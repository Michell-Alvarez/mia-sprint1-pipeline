import os, yaml, torch, random, numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import argparse
from models.cnn3d_model_solid import ViolenceDetector, ViolenceDataset, cargar_datos_desde_directorio

# === MLflow utils ===
import mlflow
from tools.mlflow_utils import (
    set_tracking,
    start_run,
    log_yaml_as_params_and_artifact,
    log_artifacts_safely,
    log_model_file,
)

# Crear un único parser
parser = argparse.ArgumentParser()

# Agregar ambos argumentos al mismo parser
parser.add_argument('--mode', type=str, default='fe_off', choices=['fe_off', 'fe_on'],
                    help="Modo de ejecución: con o sin feature engineering")
parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'solid'],
                    help="Modelo a ejecutar: baseline o solid")
parser.add_argument('--fe', type=str, default='1', choices=['0', '1', '2', '3'],
                    help="Tipo de FE, 0: sin features | 1: combined_features | 2: color_features | 3: lbp_features")
                    
# Parsear argumentos una sola vez
args = parser.parse_args()

# Construir la ruta del archivo de configuración
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'configs', f'config_{args.model}_{args.mode}.yaml')

# ---------- Utilidades ----------
def set_seed(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_run_dir(config, model_fe:str):
    """
    Prioriza:
    1) config['paths']['run_root'] si viene desde el entrenamiento,
    2) config['paths']['checkpoints'] (sube a la carpeta del run),
    3) symlink 'outputs/<model_name>/latest',
    4) último directorio por timestamp en 'outputs/<model_name>/*'
    """

    # 1) run_root
    raw_rr = config.get('paths', {}).get('run_root', None)
    if raw_rr:
        rr = Path(raw_rr)
        if rr.exists():
            print(f"[INFO] Usando run_root definido: {rr}")
            return rr
        else:
            print(f"[WARN] run_root definido pero no existe: {rr}")
            
    # 2)
    raw_rrr = config.get('paths', {}).get('checkpoints', None)
    if raw_rrr:
        rr = Path(raw_rrr)
        if rr.exists():
            print(f"[INFO] Usando checkpoints definido: {rr}")
            return rr
        else:
            print(f"[WARN] checkpoints definido pero no existe: {rr}")
            
    # 3)
    base_out = Path(config.get('paths', {}).get('base_outputs', 'outputs'))
    latest = base_out / model_fe / "latest"
    if latest.exists():
        return latest.resolve()

    # 4)
    model_out = base_out / model_fe
    if model_out.exists():
        runs = sorted([p for p in model_out.iterdir() if p.is_dir()], reverse=True)
        if runs:
            return runs[0]

    raise FileNotFoundError("No se pudo resolver el run_dir. Asegúrate de pasar rutas válidas en config o crear el symlink 'latest'.")

def make_eval_dirs(run_dir:Path):
    eval_run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = run_dir / "eval" / eval_run_id
    (root / "plots").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    return eval_run_id, root

def plot_confusion_matrix(cm, classes, save_path:Path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión - Detección de Robo Armado')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    # plt.show()

# ---------- Evaluación ----------
def evaluate_model():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Experimento/Tracking desde config (con fallback)
    mlflow_cfg = config.get('mlflow', {})
    experiment_name = mlflow_cfg.get('experiment', 'RoboArmado-Eval-CNN3D–Atencion-Temporal')
    tracking_uri = mlflow_cfg.get('tracking_uri', None)    
    
    
    # Lee semilla y modelo desde config
    seed = int(config.get('experiment', {}).get('seed', 42))
    model_name = config.get('experiment', {}).get('model', 'solid')  # 'baseline' | 'solid' ...
    mode_fe = config.get('experiment', {}).get('mode', 'fe_off')  # "fe_off" o "fe_on"
    fe = args.fe
    model_fe=model_name+'_'+mode_fe+'_'+fe

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Resolver carpeta del run a evaluar
    run_dir = resolve_run_dir(config, model_fe)
    checkpoints_dir = run_dir / "models"
    best_model_path = checkpoints_dir / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"No se encontró {best_model_path}. Verifica el run que intentas evaluar.")
    # Crear carpeta de evaluación
    eval_run_id, eval_root = make_eval_dirs(run_dir)
    plots_dir = eval_root / "plots"
    metrics_dir = eval_root / "metrics"
    # Cargar modelo y pesos
    model = ViolenceDetector().to(device)
    state_dict = torch.load(str(best_model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Dataset de test (sin aleatoriedad)
    val_paths, val_labels = cargar_datos_desde_directorio(config['data']['test_dir'])
    
    #test_dataset = ViolenceDataset(config['data']['test_dir'])
    test_dataset = ViolenceDataset(val_paths, val_labels)
    test_loader = DataLoader(test_dataset,
                             batch_size=config['training']['batch_size'],
                             shuffle=False)

    # === Inicializa MLflow y arranca run de EVALUACIÓN ===
    set_tracking(experiment_name=experiment_name, tracking_uri=tracking_uri)
    run_name = f"eval_{model_fe}_{eval_run_id}"

    with start_run(run_name=run_name, tags={
        "phase": "evaluation",
        "eval_run_id": eval_run_id,
        "train_run_id": run_dir.name,
        "model": model_name,
        "mode": mode_fe,
        "fe": fe,
        "device": str(device),
    }):
        # Guarda config como artefacto y loguea parámetros (aplanados)
        log_yaml_as_params_and_artifact(config_path, artifact_subdir="configs")


        # Evaluación
        all_preds, all_labels, all_probabilities = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probabilities = np.array(all_probabilities)

        # Métricas
        class_names = ['No Robo', 'Robo Armado']
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision, recall_w, f1_w, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Reporte por clase (para consola)
        report = classification_report(all_labels, all_preds, target_names=class_names)
        print("\n=== MATRIZ DE CONFUSIÓN ===")
        print(cm)
        print("\n=== MÉTRICAS DETALLADAS ===")
        print(report)
        print(f"Exactitud: {accuracy:.4f} | F1 (weighted): {f1_w:.4f} | Precisión (weighted): {precision:.4f}")
        print(f"Sensibilidad: {sensitivity:.4f} | Especificidad: {specificity:.4f}")
        print(f"[INFO] model={model_fe} seed={seed} run={run_dir.name} eval_run={eval_run_id}")

        # ---- Log de métricas a MLflow ----
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("f1_weighted", float(f1_w))
        mlflow.log_metric("precision_weighted", float(precision))
        mlflow.log_metric("recall_weighted", float(recall_w))
        mlflow.log_metric("specificity", float(specificity))
        mlflow.log_metric("sensitivity", float(sensitivity))

        # Tags adicionales útiles
        mlflow.set_tag("class_names", ",".join(class_names))
        mlflow.set_tag("eval_folder", str(eval_root))

        # ---- Artefactos locales y subida a MLflow ----
        # Matriz de confusión (PNG)
        cm_png = plots_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, class_names, save_path=cm_png)
        log_artifacts_safely(cm_png, artifact_path="plots")

        # CSV detallado
        detailed_df = pd.DataFrame({
            'real': all_labels,
            'predicho': all_preds,
            'probabilidad_robo': all_probabilities[:, 1],
            'probabilidad_no_robo': all_probabilities[:, 0]
        })
        detailed_csv = metrics_dir / "detailed_results.csv"
        detailed_df.to_csv(detailed_csv, index=False)
        log_artifacts_safely(detailed_csv, artifact_path="metrics")

        # CSV resumen
        metrics_df = pd.DataFrame([{
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall_w,
            'f1_weighted': f1_w,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'seed': seed,
            'model_name': model_fe,
            'train_run_id': run_dir.name,
            'eval_run_id': eval_run_id
        }])
        metrics_csv = metrics_dir / "model_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        log_artifacts_safely(metrics_csv, artifact_path="metrics")

        # También sube el modelo evaluado como referencia
        log_model_file(best_model_path, artifact_subdir="evaluated_model")

        print(f"\n✅ Guardado en disco y logueado en MLflow:\n  {cm_png}\n  {detailed_csv}\n  {metrics_csv}")

        # Devuelve algo por si quieres usarlo como módulo
        return {
            'cm': cm.tolist(),
            'metrics': metrics_df.to_dict(orient='records')[0]
        }

def analyze_misclassifications(run_dir:Path):
    """Analiza errores usando el CSV del eval más reciente de ese run"""
    eval_root = run_dir / "eval"
    if not eval_root.exists():
        print("No hay evaluaciones registradas para este run.")
        return
    last_eval = sorted([p for p in eval_root.iterdir() if p.is_dir()], reverse=True)[0]
    detailed_csv = last_eval / "metrics" / "detailed_results.csv"
    if not detailed_csv.exists():
        print("No se encontró detailed_results.csv en el eval más reciente.")
        return

    results_df = pd.read_csv(detailed_csv)
    false_positives = results_df[(results_df['real'] == 0) & (results_df['predicho'] == 1)]
    false_negatives = results_df[(results_df['real'] == 1) & (results_df['predicho'] == 0)]

    print(f"\n=== ANÁLISIS DE ERRORES ({last_eval.name}) ===")
    print(f"Falsos Positivos: {len(false_positives)}")
    if len(false_positives) > 0:
        print(f"  Prob. robo (prom): {false_positives['probabilidad_robo'].mean():.3f}  "
              f"rango: [{false_positives['probabilidad_robo'].min():.3f}, {false_positives['probabilidad_robo'].max():.3f}]")

    print(f"Falsos Negativos: {len(false_negatives)}")
    if len(false_negatives) > 0:
        print(f"  Prob. robo (prom): {false_negatives['probabilidad_robo'].mean():.3f}  "
              f"rango: [{false_negatives['probabilidad_robo'].min():.3f}, {false_negatives['probabilidad_robo'].max():.3f}]")

if __name__ == "__main__":
    out = evaluate_model()
    # Reutiliza el mismo run_dir resuelto para analizar el eval más reciente
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    # 👇 Antes pasabas solo 'model'; aquí debe ser el mismo 'model_fe' usado arriba
    model_name = cfg.get('experiment', {}).get('model', args.model)
    mode_fe = cfg.get('experiment', {}).get('mode', args.mode)
    fe = args.fe
    model_fe = f"{model_name}_{mode_fe}_{fe}"
    rd = resolve_run_dir(cfg, model_fe)
    analyze_misclassifications(rd)    
    '''
    # Reutiliza el mismo run_dir resuelto para analizar el eval más reciente
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    rd = resolve_run_dir(cfg, cfg.get('experiment', {}).get('model', 'solid'))
    analyze_misclassifications(rd)
    '''
