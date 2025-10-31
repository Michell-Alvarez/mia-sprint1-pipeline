# models/train_cnn3d_baseline.py
import os
import yaml
import torch, random, numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
from pathlib import Path
from datetime import datetime
import argparse

from cnn3d_model_baseline import Baseline3DCNN, VideoDataset
# Si realmente está en models.cnn3d_model_baseline, usa:
# from models.cnn3d_model_baseline import Baseline3DCNN, VideoDataset

import mlflow
import mlflow.pytorch  # opcional: por si luego quieres usar autolog para PyTorch

# ------------------ Parser ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='fe_off', choices=['fe_off', 'fe_on'],
                    help="Modo de ejecución: con o sin feature engineering")
parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'solid'],
                    help="Modelo a ejecutar: baseline o solid")
parser.add_argument('--fe', type=str, default='1', choices=['0', '1', '2', '3'],
                    help="Tipo de FE, 0: sin features | 1: combined_features | 2: color_features | 3: lbp_features")
args = parser.parse_args()

# ------------------ Config path ------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'configs', f'config_{args.model}_{args.mode}.yaml')

# ------------------ Utils ------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_run_dirs(model_fe: str, base_out: str = "outputs"):
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(base_out) / model_fe / run_id
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    return run_id, root

def setup_logging(log_file: Path):
    # Fuerza handlers explícitos (evita que basicConfig sea ignorado si ya hay handlers)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Limpia handlers previos
    for h in list(logger.handlers):
        logger.removeHandler(h)
    # Archivo
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    # Consola
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger

# ------------------ Main ------------------
def main():
    # Cargar config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # MLflow tracking (usa var de entorno si existe; si no, local ./mlruns)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get('experiment', {}).get('name', 'RoboArmado – CNN3D'))

    # Parámetros de experimento
    model_name = config.get('experiment', {}).get('model', args.model)   # baseline|solid
    mode_fe    = config.get('experiment', {}).get('mode', args.mode)     # fe_off|fe_on
    fe         = args.fe
    model_fe   = f"{model_name}_{mode_fe}_{fe}"
    seed       = int(config.get('experiment', {}).get('seed', 42))
    set_seed(seed)

    # Crear run_id y carpetas de salida
    run_id, run_root = make_run_dirs(model_fe, base_out=config.get('paths', {}).get('base_outputs', 'outputs'))

    # Actualizar rutas derivadas en memoria
    paths = config.setdefault('paths', {})
    paths['run_root']        = str(run_root)
    paths['logs']            = str(run_root / "logs" / "training_log.txt")
    paths['checkpoints_dir'] = str(run_root / "models")
    paths['best_model']      = str(run_root / "models" / "best_model.pth")
    paths['metrics_dir']     = str(run_root / "metrics")

    # Abrir un run de MLflow y abarcar TODO el entrenamiento y logging
    tags = {
        "variant": f"{model_name}_{mode_fe}",      # baseline_fe_off | solid_fe_on
        "model_type": model_name,                  # baseline | solid
        "feature_engineering": "on" if mode_fe == "fe_on" else "off",
        "fe_kind": fe,                             # 0|1|2|3
        "stage": "train",
        "run_id_fs": run_id                        # carpeta física del run en outputs
    }
    with mlflow.start_run(run_name=f"{model_name}_{mode_fe}_{fe}", tags=tags):
        # Log de parámetros clave y el YAML como artefacto
        mlflow.log_param("seed", seed)
        mlflow.log_param("batch_size",   config['training']['batch_size'])
        mlflow.log_param("learning_rate",config['training']['learning_rate'])
        mlflow.log_param("epochs",       config['training']['epochs'])
        mlflow.log_artifact(config_path, artifact_path="configs")

        # Logger (asegura que el archivo se cree)
        logger = setup_logging(Path(paths['logs']))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Iniciando entrenamiento en {device} | model={model_fe} | seed={seed} | run_id={run_id}")
        print(f"[INFO] model={model_fe} seed={seed} run_id={run_id} out={paths['run_root']}")

        # Datasets (split reproducible)
        full_train_dataset = VideoDataset(config['paths']['train_index'])
        train_size = int(0.8 * len(full_train_dataset))
        val_size   = len(full_train_dataset) - train_size
        g_split    = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=g_split)

        # DataLoaders (shuffle reproducible por época si quieres “anclarlo”)
        g_loader = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            generator=g_loader,
            worker_init_fn=lambda wid: np.random.seed(seed + wid)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )

        # Modelo + Opt + Criterio
        model = Baseline3DCNN().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

        # Entrenamiento
        best_accuracy = 0.0
        patience_counter = 0
        logger.info("epoch,train_loss,train_acc,val_loss,val_acc")

        for epoch in range(config['training']['epochs']):
            # --- train ---
            model.train()
            run_loss, correct, total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                run_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            train_loss = run_loss / len(train_loader)
            train_acc  = correct / total

            # --- val ---
            model.eval()
            v_loss, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    v_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    v_total += labels.size(0)
                    v_correct += (pred == labels).sum().item()
            val_loss = v_loss / len(val_loader)
            val_acc  = v_correct / v_total

            logger.info(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}")
            print(f"Epoch {epoch+1}: Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

            # MLflow métricas por época (step=epoch)
            mlflow.log_metric("train_loss", float(train_loss), step=epoch+1)
            mlflow.log_metric("train_acc",  float(train_acc),  step=epoch+1)
            mlflow.log_metric("val_loss",   float(val_loss),   step=epoch+1)
            mlflow.log_metric("val_acc",    float(val_acc),    step=epoch+1)

            # Checkpoint
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                patience_counter = 0
                Path(paths['checkpoints_dir']).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), paths['best_model'])
            else:
                patience_counter += 1

            if patience_counter >= config['training']['early_stop_patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Métrica final y artefactos del entrenamiento (dentro del run)
        mlflow.log_metric("best_val_acc", float(best_accuracy))

        logger.info(f"best_val_acc,{best_accuracy:.4f}")
        # Escribir y subir resumen de experimento
        summary_csv = Path(paths['metrics_dir']) / "experiment_summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write("model_fe,run_id,seed,best_val_acc\n")
            f.write(f"{model_fe},{run_id},{seed},{best_accuracy:.4f}\n")
        if summary_csv.exists():
            mlflow.log_artifact(str(summary_csv), artifact_path="train")

        # Subir logs y el mejor modelo
        log_file = Path(paths['logs'])
        if log_file.exists():
            mlflow.log_artifact(str(log_file), artifact_path="logs")
        best_model_pth = Path(paths['best_model'])
        if best_model_pth.exists():
            mlflow.log_artifact(str(best_model_pth), artifact_path="models")

if __name__ == "__main__":
    main()
