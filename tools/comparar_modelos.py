#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import io
import base64
import textwrap
import matplotlib.pyplot as plt

VARIANTS = ["baseline_fe_off", "baseline_fe_on", "solid_fe_off", "solid_fe_on"]

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

def is_timestamp_folder(name: str) -> bool:
    return bool(TIMESTAMP_RE.match(name))

def latest_timestamp_folder(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    candidates = [p for p in parent.iterdir() if p.is_dir() and is_timestamp_folder(p.name)]
    if not candidates:
        return None
    # ordenar por nombre (ISO-like => orden lexicográfico = cronológico)
    return sorted(candidates)[-1]

def latest_eval_run(run_root: Path) -> Path | None:
    """Dentro de <variant>/<run>/eval/<eval_run>/... retorna la más reciente si existe."""
    eval_dir = run_root / "eval"
    if not eval_dir.exists():
        return None
    return latest_timestamp_folder(eval_dir)

def read_metrics_csv(csv_path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path)
        # si tiene múltiples filas (por épocas), tomamos la última como resumen
        if len(df) > 1:
            df = df.tail(1).reset_index(drop=True)
        # redondeo a 2 decimales si es numérico (preferencia del usuario)
        for c in df.select_dtypes(include="number").columns:
            df[c] = df[c].round(2)
        return df
    except Exception:
        return None

def parse_experiment_log(log_path: Path | None) -> dict:
    """Lee experiment_summary.csv y devuelve un dict {columna: valor}."""
    data = {}
    if not log_path or not isinstance(log_path, Path) or not log_path.exists():
        return data
    try:
        df = pd.read_csv(log_path)
        # Si hay más de una fila, tomamos la última (por ejemplo, resumen final)
        if len(df) > 1:
            df = df.tail(1).reset_index(drop=True)
        # Convertir a dict: columna -> valor
        data = {col: df[col].iloc[0] for col in df.columns}
    except Exception:
        # fallback: intentar como texto si no abre como CSV
        try:
            txt = log_path.read_text(encoding="utf-8", errors="ignore")
            for line in txt.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    data[k.strip()] = v.strip()
        except Exception:
            pass
    return data

def encode_image_base64(img_path: Path) -> str | None:
    if not img_path or not img_path.exists():
        return None
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None

def find_artifacts_for_variant(base_dir: Path, variant: str) -> dict:
    variant_dir = base_dir / variant
    last_train_run = latest_timestamp_folder(variant_dir)
    if not last_train_run:
        return {"variant": variant}

    # Preferir métricas del último eval run
    eval_run = latest_eval_run(last_train_run)
    if eval_run:
        metrics_csv = eval_run / "metrics" / "model_metrics.csv"
        cm_png = eval_run / "plots" / "confusion_matrix.png"
    else:
        metrics_csv = last_train_run / "metrics" / "model_metrics.csv"
        cm_png = last_train_run / "plots" / "confusion_matrix.png"

    log_path = last_train_run / "metrics" / "experiment_summary.csv"

    return {
        "variant": variant,
        "train_run_path": last_train_run,
        "eval_run_path": eval_run,
        "metrics_csv": metrics_csv if metrics_csv and metrics_csv.exists() else None,
        "confusion_png": cm_png if cm_png and cm_png.exists() else None,
        "log_path": log_path if log_path.exists() else None,
    }

   
def unify_columns(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une por filas con outer join de columnas, pero sin incluir la columna 'variant'."""
    rows = []
    for variant, df in dfs.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, "variant", variant)
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    
    out = pd.concat(rows, axis=0, ignore_index=True)

    # ordenar columnas: métricas comunes (accuracy/precision/recall/f1...), luego el resto
    cols = [c for c in out.columns if c != "variant"]
    def has(col, pat_list):
        lc = col.lower()
        return any(p in lc for p in pat_list)

    key_order = []
    for c in cols:
        if has(c, ["acc"]): key_order.append((0, c))
        elif has(c, ["f1"]): key_order.append((1, c))
        elif has(c, ["precision"]): key_order.append((2, c))
        elif has(c, ["recall", "tpr", "sensitivity"]): key_order.append((3, c))
        elif has(c, ["specificity", "tnr"]): key_order.append((4, c))
        elif has(c, ["auc", "roc"]): key_order.append((5, c))
        elif has(c, ["loss", "error"]): key_order.append((6, c))
        else: key_order.append((9, c))

    ordered = [c for _, c in sorted(key_order, key=lambda x: (x[0], x[1]))]
    out = out.reindex(columns=ordered)
    return out


def make_cm_grid_image(cm_paths: list[tuple[str, Path]], out_path: Path) -> Path | None:
    """Crea una grilla 2x2 con las confusion_matrix.png disponibles."""
    present = [(name, p) for name, p in cm_paths if p and p.exists()]
    if not present:
        return None
    # armar figura hasta 2x2
    n = min(4, len(present))
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        name, p = present[i]
        img = Image.open(p).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(name, fontsize=11)
        axes[i].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path

# --------- Utilidades de plotting ---------
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def _annotate_bars(ax):
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f"{height:.2f}", 
                        (p.get_x() + p.get_width()/2.0, height),
                        ha='center', va='bottom', fontsize=9, rotation=0)

def _plot_metrics_heatmap(fig_path: Path, df: pd.DataFrame, index_col: str = "model_name"):
    candidate_metrics = [
        "accuracy", "f1_weighted", "precision_weighted",
        "recall_weighted", "sensitivity", "specificity"
    ]
    present = [m for m in candidate_metrics if m in df.columns]
    if not present or index_col not in df.columns:
        return False

    df_plot = df[[index_col] + present].copy()
    for c in present: df_plot[c] = df_plot[c].map(_safe_float)
    if df_plot[present].isna().all().all(): return False

    models = df_plot[index_col].astype(str).tolist()
    data = df_plot[present].values.astype(float)

    fig, ax = plt.subplots(figsize=(1.2 * len(present) + 3, 0.8 * len(models) + 2))
    im = ax.imshow(data, aspect="auto")
    ax.set_xticks(np.arange(len(present)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_yticklabels(models)
    ax.set_title("Heatmap de métricas por modelo")

    for i in range(len(models)):
        for j in range(len(present)):
            val = data[i, j]
            ax.text(j, i, "" if np.isnan(val) else f"{val:.2f}", ha="center", va="center")

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Valor", rotation=90, va="center")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True

def _plot_metrics_bars(fig_path: Path, df: pd.DataFrame, index_col: str, metric_cols: list, title: str):
    # Filtrar columnas presentes
    present = [c for c in metric_cols if c in df.columns]
    if not present or index_col not in df.columns:
        return False

    plot_df = df[[index_col] + present].copy()
    for c in present: plot_df[c] = plot_df[c].map(_safe_float)
    if plot_df[present].isna().all().all(): return False

    # Parámetros de barras agrupadas
    x = np.arange(len(plot_df[index_col]))
    n_groups = len(present)
    width = min(0.8 / max(1, n_groups), 0.22)  # ancho por serie

    fig, ax = plt.subplots(figsize=(1.6 * len(plot_df[index_col]) + 3, 4.8))
    for i, m in enumerate(present):
        ax.bar(x + i*width - (width*(n_groups-1)/2), plot_df[m].values, width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[index_col].astype(str), rotation=20, ha="right")
    ax.set_ylim(0.0, max(1.0, np.nanmax(plot_df[present].values) * 1.1))
    ax.set_ylabel("Valor (0-1)")
    ax.set_title(title)
    ax.legend()
    _annotate_bars(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True

# --- Añade este helper junto a los otros (arriba) ---
def _autodetect_train_metrics(df: pd.DataFrame, label_col: str) -> list[str]:
    """
    Elige columnas métricas para graficar barras de entrenamiento:
    - Prioriza un set conocido si existen
    - Si no, toma columnas numéricas (o casteables) excluyendo seed/ids/label
    """
    candidates = [
        # pérdidas/accuracies típicas
        "train_loss","val_loss","valid_loss","train_acc","train_accuracy","val_acc","valid_acc",
        "train_f1","val_f1","train_precision","val_precision","train_recall","val_recall",
        "train_f1_weighted","val_f1_weighted","train_precision_weighted","val_precision_weighted",
        # tu métrica clave
        "best_val_acc",
    ]
    present = [c for c in candidates if c in df.columns]

    if present:
        return present

    # Si no encontró candidatos "típicos", auto-detectar métricas numéricas
    # Incluye columnas casteables a float y excluye seed/ids/label
    excl = {label_col.lower(), "seed", "run_id", "train_run_id", "eval_run_id"}
    metrics = []
    for c in df.columns:
        cl = c.lower()
        if cl in excl or cl.endswith("_id"):
            continue
        # intentar castear a float
        try:
            arr = pd.to_numeric(df[c], errors="coerce")
            if arr.notna().any():
                metrics.append(c)
        except Exception:
            pass
    return metrics

    
# --------- Reporte principal ---------
def build_html_report(dst_dir: Path, table: pd.DataFrame, cm_paths: dict[str, Path], logs: dict[str, dict]):
    dst_dir.mkdir(parents=True, exist_ok=True)
    html_rows = []

    # Título
    html_rows.append("<h2>Comparativa de Modelos</h2>")
    html_rows.append(f"<p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

    # =========================
    # TEST: tabla + gráficos
    # =========================
    html_rows.append("<h3>Resumen de experimentos en el test (model_metrics.csv)</h3>")

    # Tabla test
    if isinstance(table, pd.DataFrame) and not table.empty:
        tbl = table.copy()
        html_rows.append(tbl.to_html(index=False, border=0, float_format=lambda x: f"{x:.2f}", escape=False))
    else:
        html_rows.append("<p><em>No se encontraron métricas para construir la tabla.</em></p>")

    # Gráfico de barras (test)
    if isinstance(table, pd.DataFrame) and not table.empty:
        bars_test_path = dst_dir / "metrics_bars_test.png"
        test_metrics = ["accuracy", "f1_weighted", "precision_weighted",
                        "recall_weighted", "sensitivity", "specificity"]
        ok_bars_test = _plot_metrics_bars(
            bars_test_path, table, index_col="model_name",
            metric_cols=test_metrics, title="Barras comparativas (Test)"
        )
        html_rows.append("<h3>Visualización comparativa (Test)</h3>")
        if ok_bars_test and bars_test_path.exists():
            html_rows.append(f'<div><img src="{bars_test_path.name}" alt="Barras Test" style="max-width:100%;height:auto;"></div>')
        else:
            html_rows.append("<p><em>No se pudo generar el gráfico de barras del test.</em></p>")

        # Heatmap (opcional, ya integrado)
        heat_path = dst_dir / "metrics_heatmap.png"
        ok_heat = _plot_metrics_heatmap(heat_path, table, index_col="model_name")
        if ok_heat and heat_path.exists():
            html_rows.append(f'<div style="margin-top:8px;"><img src="{heat_path.name}" alt="Heatmap Test" style="max-width:100%;height:auto;"></div>')

    # =========================
    # TRAIN: tabla + gráfico
    # =========================
    html_rows.append("<h3>Resumen de experimentos en el entrenamiento (experiment_summary.csv)</h3>")

    # Armar DF de logs (y mantener oculto 'variant' en tabla)
    rows = []
    for variant, kv in (logs or {}).items():
        kv = kv or {}
        row = {"variant": variant}
        row.update(kv)
        rows.append(row)

    if rows:
        df_logs = pd.DataFrame(rows)

        # Recortar textos
        def trim_text(x, width=200):
            if pd.isna(x): return ""
            s = str(x)
            return s if len(s) <= width else textwrap.shorten(s, width=width, placeholder="…")

        obj_cols = [c for c in df_logs.columns if df_logs[c].dtype == "object"]
        for c in obj_cols: df_logs[c] = df_logs[c].map(trim_text)

        # Orden sugerido
        def rank(col: str) -> tuple:
            cl = col.lower()
            if cl == "variant": return (0, col)
            if any(k in cl for k in ["epoch", "epochs"]):      return (1, col)
            if cl in ("lr", "learning_rate"):                  return (2, col)
            if "batch" in cl:                                  return (3, col)
            if any(k in cl for k in ["optimizer", "sched"]):   return (4, col)
            if "seed" in cl:                                   return (5, col)
            if any(k in cl for k in ["model", "backbone"]):    return (6, col)
            if "train_loss" in cl:                             return (10, col)
            if "val_loss" in cl or "valid_loss" in cl:         return (11, col)
            if "test_loss" in cl:                              return (12, col)
            if cl == "loss":                                   return (13, col)
            if any(k in cl for k in ["train_acc", "train_accuracy"]):   return (20, col)
            if any(k in cl for k in ["val_acc", "valid_acc"]):          return (21, col)
            if any(k in cl for k in ["test_acc", "accuracy"]):          return (22, col)
            if "f1" in cl:                                              return (23, col)
            if "precision" in cl:                                       return (24, col)
            if any(k in cl for k in ["recall", "tpr", "sensitivity"]):  return (25, col)
            if any(k in cl for k in ["specificity", "tnr"]):            return (26, col)
            if any(k in cl for k in ["auc", "roc"]):                    return (27, col)
            if any(k in cl for k in ["time", "duration", "elapsed"]):   return (30, col)
            if any(k in cl for k in ["start", "end", "date", "timestamp"]): return (31, col)
            if any(k in cl for k in ["run_id", "uuid"]):                return (32, col)
            return (50, col)

        # Ordenar columnas y ocultar 'variant'
        ordered_cols = [c for c in sorted(df_logs.columns, key=rank) if c.lower() != "variant"]
        df_logs = df_logs.reindex(columns=ordered_cols)

        # Tabla: formatear números a 2 decimales excepto seed
        num_cols = df_logs.select_dtypes(include="number").columns
        formatters = {}
        for c in num_cols:
            if "seed" not in c.lower():
                formatters[c] = lambda v, _c=c: "" if pd.isna(v) else f"{v:.2f}"
        html_rows.append(df_logs.to_html(index=False, border=0, escape=True, formatters=formatters))

        # ---------- Gráfico de barras (train) ----------
        # Etiqueta eje X: preferir 'model_fe', si no 'model_name', si no creamos una
        if "model_fe" in df_logs.columns:
            label_col = "model_fe"
        elif "model_name" in df_logs.columns:
            label_col = "model_name"
        else:
            df_logs["_label"] = [f"exp_{i+1}" for i in range(len(df_logs))]
            label_col = "_label"

        # Detectar métricas disponibles (soporta 'best_val_acc')
        train_metrics = _autodetect_train_metrics(df_logs, label_col)
        bars_train_path = dst_dir / "metrics_bars_train.png"

        ok_bars_train = False
        if train_metrics:
            # Renombrar temporalmente la columna de etiqueta para el helper genérico
            ok_bars_train = _plot_metrics_bars(
                bars_train_path,
                df_logs.rename(columns={label_col: "LABEL_FOR_PLOT"}),
                index_col="LABEL_FOR_PLOT",
                metric_cols=train_metrics,
                title="Barras comparativas (Entrenamiento/Validación)"
            )

        html_rows.append("<h3>Visualización comparativa (Entrenamiento)</h3>")
        if ok_bars_train and bars_train_path.exists():
            html_rows.append(
                f'<div><img src="{bars_train_path.name}" alt="Barras Entrenamiento" style="max-width:100%;height:auto;"></div>'
            )
        else:
            html_rows.append("<p><em>No se pudo generar el gráfico de barras del entrenamiento (faltan columnas métricas).</em></p>")
    else:
        html_rows.append("<p><em>No se hallaron logs.</em></p>")

    # Guardar HTML
    html = "\n".join(html_rows)
    #(dst_dir / "comparativa.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Comparador de runs para 4 variantes.")
    parser.add_argument("--base-dir", type=str, default="outputs", help="Directorio base de outputs/")
    parser.add_argument("--variants", type=str, nargs="*", default=VARIANTS, help="Lista de variantes a comparar")
    parser.add_argument("--prefer-eval", action="store_true", default=True, help="(por defecto True) Usar métricas de eval si existen")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_dir / "_comparaciones" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    for v in args.variants:
        artifacts[v] = find_artifacts_for_variant(base_dir, v)

    # cargar métricas CSV
    metrics_by_variant = {}
    for v, art in artifacts.items():
        df = None
        p = art.get("metrics_csv")
        if p:
            df = read_metrics_csv(p)
        metrics_by_variant[v] = df

    table = unify_columns(metrics_by_variant)
    
    # Reordenar columnas del DataFrame "table"
    desired_order = [
        "model_name", "train_run_id", "eval_run_id", "seed", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "sensitivity", "specificity"
    ]

    # Mantener las columnas existentes y en ese orden
    ordered_cols = [c for c in desired_order if c in table.columns] + \
                   [c for c in table.columns if c not in desired_order]

    table = table.reindex(columns=ordered_cols)


    # agregar columna con ruta del run (para rastreabilidad)
    run_col = []
    for v in table["variant"] if not table.empty and "variant" in table.columns else []:
        run_path = artifacts[v].get("eval_run_path") or artifacts[v].get("train_run_path")
        run_col.append(str(run_path) if run_path else "")
    if run_col:
        table.insert(1, "run_path", run_col)

    # parsear logs
    logs = {v: parse_experiment_log(artifacts[v].get("log_path")) for v in artifacts}

    # guardar CSV
    csv_path = out_dir / "experimento_prueba_consolidado.csv"
    if not table.empty:
        table.to_csv(csv_path, index=False)

    # Consolidar logs experiment_summary.csv
    logs_rows = []
    for variant, data in logs.items():
        if data:
            row = {"variant": variant}
            row.update(data)
            logs_rows.append(row)

    df_logs = pd.DataFrame(logs_rows)

    # Filtrar solo columnas solicitadas si existen en el DataFrame
    cols_keep = ["variant", "run_id", "best_val_acc", "seed"]
    cols_present = [c for c in cols_keep if c in df_logs.columns]
    df_logs = df_logs[cols_present]

    rename_map = {
        "variant": "model_name",
        "run_id": "train_run_id",
        "best_val_acc": "best_val_acc",
        "seed": "seed"
    }

    df_logs = df_logs.rename(columns=rename_map)

    logs_csv_path = out_dir / "experimento_entrenamiento_consolidado.csv"
    if not df_logs.empty:
        df_logs.to_csv(logs_csv_path, index=False)


    # generar grilla de CM opcional
    cm_paths = {v: artifacts[v].get("confusion_png") for v in artifacts}
    grid_path = out_dir / "grid_confusion_matrices.png"
    grid_done = make_cm_grid_image(list(cm_paths.items()), grid_path)

    # reporte HTML embebiendo imágenes individuales; además el PNG de grilla (si existe) puede verse aparte
    build_html_report(out_dir, table, cm_paths, logs)

    print("==> Comparativa generada en:")
    print(f"- CSV:   {csv_path if csv_path.exists() else 'No creado'}")
    print(f"- HTML:  {(out_dir / 'comparativa.html')}")
    print(f"- GRID:  {grid_done if grid_done else 'No se generó (no había imágenes)'}")

if __name__ == "__main__":
    main()

