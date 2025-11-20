# optimize.py
import os
import json
import yaml
import argparse
from pathlib import Path
from types import SimpleNamespace

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
import math
from optuna.exceptions import TrialPruned

# ==========================================================
# Importa tu trainer (asegúrate de que no haga parse_args() al importar)
# ==========================================================
import train_cnn3d_solid as trainer
import csv

# ==========================================================
# Funciones auxiliares
# ==========================================================

def _is_finite(x) -> bool:
    try:
        xf = float(x)
        return math.isfinite(xf)
    except Exception:
        return False

def _read_metric_from_csv(csv_path: Path, prefer_keys: list[str]) -> tuple[str, float] | None:
    """
    Lee la PRIMERA clave existente en prefer_keys desde el CSV y devuelve (nombre, valor).
    Retorna None si no encuentra ninguna o si el valor no es finito.
    """
    
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            last_row = row
        if last_row is None:
            return None
        for k in prefer_keys:
            if k in last_row and _is_finite(last_row[k]):
                return (k, float(last_row[k]))
    return None
    
def save_optuna_plots(study, out_dir: Path, top_k: int = 10):
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CSV top-k ---
    trials_sorted = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None],
        key=lambda t: t.value,
        reverse=(study.direction.name == "MAXIMIZE")
    )
    # ... (tu CSV igual que antes) ...

    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
    )
    from optuna.importance import MeanDecreaseImpurityImportanceEvaluator

    figs = {}
    # Siempre seguro:
    try:
        figs["optimization_history"] = plot_optimization_history(study)
    except Exception as e:
        print(f"[WARN] optimization_history falló: {e}")

    # Estos requieren más datos; rodea con try/except y fallback a MDI
    for name, fn in {
        "parallel_coordinate": plot_parallel_coordinate,
        "slice": plot_slice,
    }.items():
        try:
            figs[name] = fn(study)
        except Exception as e:
            print(f"[WARN] {name} falló: {e}")

    # Importances: intenta fANOVA y cae a MDI si falla
    try:
        figs["param_importances"] = plot_param_importances(study)
    except Exception as e:
        print(f"[WARN] fANOVA falló ({e}). Probando MDI…")
        try:
            figs["param_importances"] = plot_param_importances(
                study, evaluator=MeanDecreaseImpurityImportanceEvaluator()
            )
        except Exception as e2:
            print(f"[WARN] MDI también falló: {e2}")

    # Guardar lo que haya salido
    for name, fig in figs.items():
        if fig is None:
            continue
        fig.write_html(str(out_dir / f"{name}.html"), include_plotlyjs="cdn")
        try:
            import kaleido  # noqa: F401
            fig.write_image(str(out_dir / f"{name}.png"))
        except Exception as e:
            print(f"[WARN] No PNG para {name}: {e}")



def build_sampler(kind: str, seed: int = 42):
    """Crea el muestreador de Optuna."""
    k = kind.lower()
    if k == "tpe":
        return TPESampler(seed=seed)
    elif k == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"Sampler no soportado: {kind}")


def build_pruner(kind: str, warmup_steps=5, min_resource=5):
    """Crea el pruner de Optuna."""
    k = kind.lower()
    if k == "median":
        return MedianPruner(n_warmup_steps=warmup_steps)
    elif k == "sha":
        return SuccessiveHalvingPruner(min_resource=min_resource)
    elif k == "hyperband":
        return HyperbandPruner(min_resource=min_resource)
    else:
        raise ValueError(f"Pruner no soportado: {kind}")


def suggest_space(trial: optuna.Trial, space: dict):
    """Construye el espacio de búsqueda dinámicamente."""
    hp = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "int":
            hp[name] = trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step"))
        elif t == "float":
            hp[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif t == "logfloat":
            hp[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif t == "categorical":
            hp[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Tipo no soportado: {t}")
    return hp


# ==========================================================
# Main
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Optimización de hiperparámetros con Optuna")
    parser.add_argument("--config", "-c", type=str, default="config_solid_fe_on.yaml",
                        help="Ruta al archivo .yaml de configuración base")
    parser.add_argument("--optuna_name", type=str, default="optuna_solid_fe_on_1",
                        help="Nombre del experimento Optuna (outputs/<name>/trial_*/...)")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"],
                        help="Algoritmo de muestreo (TPE o Random)")
    parser.add_argument("--pruner", type=str, default="median", choices=["median", "sha", "hyperband"],
                        help="Tipo de pruning")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"],
                        help="Dirección de optimización (maximize=val_acc, minimize=val_loss)")
    parser.add_argument("--n_trials", type=int, default=20, help="Número de trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="solid", choices=["solid", "baseline"])
    parser.add_argument("--mode", type=str, default="fe_on", choices=["fe_on", "fe_off"])
    parser.add_argument("--fe", type=str, default="1", choices=["0", "1", "2", "3"])
    args = parser.parse_args()

    '''
    # --- Leer configuración base ---
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)
    '''
    # Si la ruta no es absoluta, buscarla relativa al directorio del script
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")

    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)


    # --- Espacio de búsqueda ---
    search_space = {
        "lr": {"type": "logfloat", "low": 1e-4, "high": 1e-2},
        "weight_decay": {"type": "logfloat", "low": 1e-6, "high": 1e-3},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
    }

    # --- Configurar directorios ---
    optuna_root = Path("outputs") / args.optuna_name
    optuna_root.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{(optuna_root / 'optuna.db').as_posix()}"

    # --- Crear estudio ---
    study = optuna.create_study(
        study_name=args.optuna_name,
        storage=storage_url,
        load_if_exists=True,
        direction=args.direction,
        sampler=build_sampler(args.sampler, seed=args.seed),
        pruner=build_pruner(args.pruner, warmup_steps=5, min_resource=5),
    )

    # ======================================================
    # Objective function
    # ======================================================
    def objective(trial: optuna.Trial):
        trial_dir = optuna_root / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        hp = suggest_space(trial, search_space)

        # --- 🔥 Nuevo bloque: convierte el batch efectivo en batch físico seguro + acumulación ---
        effective_batch = int(hp["batch_size"])

        # Mapea a batch físico compatible con tu GPU (8 GB)
        if effective_batch == 128:
            physical_batch, accum = 32, 4
        elif effective_batch == 64:
            physical_batch, accum = 32, 2
        else:  # 32 o cualquier otro
            physical_batch, accum = 32, 1


        # --- Armar los overrides con AMP activado ---
        overrides = {
            "training": {
                "lr": float(hp["lr"]),
                "weight_decay": float(hp["weight_decay"]),
                "batch_size": physical_batch,      # 👈 batch físico (entra en VRAM)
                "grad_accum_steps": accum,         # 👈 pasos de acumulación
                "amp": True                        # 👈 forzar AMP (FP16)
            },
            "paths": {"base_outputs": str(trial_dir)},
        }

        # 🔧 Evita fragmentación CUDA
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

        # Exporta los overrides a train.py
        os.environ["OPTUNA_OVERRIDES"] = json.dumps(overrides)
        os.environ["OPTUNA_TRIAL_DIR"] = str(trial_dir)
        os.environ["OPTUNA_EXPERIMENT_NAME"] = args.optuna_name
        os.environ["OPTUNA_DIRECTION"] = args.direction


        # Ejecutar entrenamiento
        train_args = SimpleNamespace(
            model=args.model,
            mode=args.mode,
            fe=args.fe,
            base_outputs=str(trial_dir),
            experiment_name=args.optuna_name,
        )

        try:
            # Tu trainer DEBE escribir métricas en trial_dir/"metrics"/"experiment_summary.csv"
            trainer.main(args=train_args, trial=trial)
        except TrialPruned:
            # Deja que Optuna cuente este trial como "pruned" (no COMPLETED con valor falso)
            raise
        except Exception as e:
            print(f"[ERROR] Trial {trial.number} falló durante entrenamiento: {e}")
            # En vez de devolver 0.0/inf, pruneamos: no contamina la distribución de valores
            raise TrialPruned()

        # === Leer métricas (búsqueda recursiva) ===
        # 1) Busca experiment_summary.csv en cualquier subcarpeta del trial
        candidates = sorted(
            trial_dir.glob("**/metrics/experiment_summary.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,  # primero el más reciente
        )

        if not candidates:
            print(f"[WARN] No se encontró experiment_summary.csv bajo {trial_dir}. Prune.")
            raise TrialPruned()

        summary_csv = candidates[0]
        print(f"[INFO] Usando métricas de: {summary_csv}")

        # 2) Define columnas preferidas según la dirección
        prefer_maximize = ["best_val_acc", "best_val_f1", "val_acc", "val_f1"]
        prefer_minimize = ["best_val_loss", "val_loss"]

        # 3) Lee la métrica desde el CSV
        if args.direction == "maximize":
            metric = _read_metric_from_csv(summary_csv, prefer_maximize)
        else:
            metric = _read_metric_from_csv(summary_csv, prefer_minimize)

        if metric is None:
            print(f"[WARN] No se encontró una métrica válida en {summary_csv}. Prune.")
            raise TrialPruned()

        metric_name, metric_value = metric

        # 4) Validación de finitud
        if not _is_finite(metric_value):
            print(f"[WARN] Métrica no finita ({metric_name}={metric_value}). Prune.")
            raise TrialPruned()

        # 5) Guarda info útil en atributos del trial
        trial.set_user_attr("metrics_csv_path", str(summary_csv))
        trial.set_user_attr("metric_name", metric_name)
        trial.set_user_attr("metric_value", float(metric_value))
        trial.set_user_attr("best_ckpt_path", str(trial_dir / "checkpoints"))
        trial.set_user_attr("effective_batch", effective_batch)
        trial.set_user_attr("physical_batch", physical_batch)
        trial.set_user_attr("grad_accum_steps", accum)

        # 6) Devuelve el valor tal cual (Maximize: mayor mejor; Minimize: menor mejor)
        return float(metric_value)
        

    # --- Ejecutar optimización ---
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    # --- Resultados finales ---
    print("\n=== RESULTADOS OPTUNA ===")
    print(f"Sampler: {args.sampler} | Pruner: {args.pruner} | Direction: {args.direction}")
    print("Best value:", study.best_value)
    print("Best params:", study.best_trial.params)
    print("Best trial dir:", optuna_root / f"trial_{study.best_trial.number}")

    plots_dir = optuna_root / "plots"
    save_optuna_plots(study, plots_dir, top_k=10)
    print(f"Gráficos y top-k guardados en: {plots_dir}")



if __name__ == "__main__":
    main()

