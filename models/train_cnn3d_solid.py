# src/train.py
import os
import yaml, logging
import torch, random, numpy as np
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import argparse

from cnn3d_model_solid import (
    build_dataloaders,
    ViolenceDetector
)

import mlflow
import mlflow.pytorch  # opcional: por si luego quieres usar autolog para PyTorch


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


# Activa una optimización en cudnn para mejorar el rendimiento en modelos estáticos
torch.backends.cudnn.benchmark = True
# Fija la semilla de PyTorch para resultados reproducibles
torch.manual_seed(42)
# Fija la semilla de NumPy para garantizar la reproducibilidad
np.random.seed(42)
seed=42

# ---------- 2) Estructura de salidas por run ----------
def make_run_dirs(model_fe: str, base_out: str = "outputs"):
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(base_out) / model_fe / run_id
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    #(root / "plots").mkdir(parents=True, exist_ok=True)
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
    
    
def main():
 
    # Cargar config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # MLflow tracking (usa var de entorno si existe; si no, local ./mlruns)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.get('experiment', {}).get('name', 'RoboArmado – CNN3D – Atencion  – Temporal'))
    
    # Lee nombre del modelo y seed desde config
    model_name = cfg.get('experiment', {}).get('model', 'solid')  # "baseline" o "solid"
    mode_fe = cfg.get('experiment', {}).get('mode', 'fe_on')  # "fe_off" o "fe_on"
    fe = args.fe
    model_fe=model_name+'_'+mode_fe+'_'+fe
    
    # Crear run_id y carpetas de salida
    run_id, run_root = make_run_dirs(model_fe, base_out=cfg.get('paths', {}).get('base_outputs', 'outputs'))

    # Actualizar rutas derivadas en memoria (sin tocar el YAML)
    paths = cfg.setdefault('paths', {})
    paths['run_root'] = str(run_root)
    paths['logs'] = str(run_root / "logs" / "training_log.txt")
    paths['checkpoints_dir'] = str(run_root / "models")
    paths['best_model'] = str(run_root / "models" / "best_model.pth")
    paths['metrics_dir'] = str(run_root / "metrics")
    #paths['plots_dir'] = str(run_root / "plots")

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
        mlflow.log_param("batch_size",   cfg['training']['batch_size'])
        mlflow.log_param("learning_rate",cfg['training']['lr'])
        mlflow.log_param("epochs",       cfg['training']['epochs'])
        mlflow.log_artifact(config_path, artifact_path="configs")

        # Logger
        logger = setup_logging(Path(paths['logs']))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Iniciando entrenamiento en {device} | model={model_fe} | seed={seed} | run_id={run_id}")
        print(f"[INFO] model={model_fe} seed={seed} run_id={run_id} out={paths['run_root']}")
        
        print(f"Dispositivo: {device}")

        # --- Datos ---
        train_loader, val_loader, train_labels, _ = build_dataloaders(cfg, args.fe)

        # Pos-weight opcional (desbalance)
        class_counts = np.bincount(train_labels) if len(train_labels) > 0 else np.array([1, 1])
        if len(class_counts) < 2:  # si falta alguna clase, evita crash
            class_counts = np.append(class_counts, 1)
        neg, pos = class_counts[0], class_counts[1]
        # Peso para clase 1 relativo a 0 (más peso si hay menos positivos)
        weights = torch.tensor([1.0, max(neg / max(pos, 1), 1.0)], dtype=torch.float32).to(device)

        # --- Modelo & criterios ---
        model = ViolenceDetector(num_classes=cfg['data']['num_classes']).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg['training'].get('label_smoothing', 0.0))
        optimizer = optim.AdamW(model.parameters(),
                                lr=cfg['training']['lr'],
                                weight_decay=cfg['training'].get('weight_decay', 1e-4))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
        #scaler = GradScaler(enabled=cfg['training'].get('amp', True))
        scaler = GradScaler()
        
        # --- Entrenamiento ---
        best_acc = 0.0
        patience = cfg['training'].get('early_stop_patience', 7)
        min_delta = cfg['training'].get('min_delta', 0.005)
        patience_counter = 0

        os.makedirs(os.path.dirname(cfg['paths']['best_model']), exist_ok=True)
        if 'log_txt' in cfg['paths']:
            os.makedirs(os.path.dirname(cfg['paths']['log_txt']), exist_ok=True)
        
        logger.info("epoch,train_loss,train_acc,val_loss,val_acc")
        for epoch in range(cfg['training']['epochs']):
        #for epoch in range(1, cfg['training']['epochs'] + 1):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            #for vids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}"):
            for vids, labels in tqdm(train_loader):
                vids, labels = vids.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=cfg['training'].get('amp', True)):
                    outputs = model(vids)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            train_acc = correct / max(total, 1)

            # --- Validación ---
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad(), autocast(enabled=cfg['training'].get('amp', True)):
                for vids, labels in val_loader:
                    vids, labels = vids.to(device), labels.to(device)
                    outputs = model(vids)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()

            val_acc = correct / max(total, 1)
            scheduler.step()

            logger.info(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}")
            print(f"Epoch {epoch+1}: Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

            # MLflow métricas por época (step=epoch)
            mlflow.log_metric("train_loss", float(train_loss), step=epoch+1)
            mlflow.log_metric("train_acc",  float(train_acc),  step=epoch+1)
            mlflow.log_metric("val_loss",   float(val_loss),   step=epoch+1)
            mlflow.log_metric("val_acc",    float(val_acc),    step=epoch+1)

            # --- Early stopping + guardar mejor ---
            improved = val_acc > best_acc + min_delta
            if improved:
                best_acc = val_acc
                patience_counter = 0
                '''
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc
                }, cfg['paths']['best_model'])
                '''            
                torch.save(model.state_dict(), paths['best_model'])
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping en epoch {epoch}")
                    break

            # Métrica final y artefactos del entrenamiento (dentro del run)
            mlflow.log_metric("best_val_acc", float(best_acc))

            # --- Log consola + archivo ---
            log_line = (f"Epoch {epoch+1:03d} | "
                        f"TrainLoss {train_loss/len(train_loader):.4f} Acc {train_acc:.4f} | "
                        f"ValLoss {val_loss/len(val_loader):.4f} Acc {val_acc:.4f} | "
                        f"Best {best_acc:.4f}")
            print(log_line)
            if 'log_txt' in cfg['paths']:
                with open(cfg['paths']['log_txt'], 'a', encoding='utf-8') as fh:
                    fh.write(log_line + "\n")
            
        print(f"Mejor Val Acc: {best_acc:.4f}")
        print(f"Checkpoint guardado en: {cfg['paths']['best_model']}")

        logger.info(f"best_val_acc,{best_acc:.4f}")
        # Escribir y subir resumen de experimento
        summary_csv = Path(paths['metrics_dir']) / "experiment_summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)  
        with open(summary_csv, "w", encoding="utf-8") as f:
            f.write("model_fe,run_id,seed,best_val_acc\n")
            f.write(f"{model_fe},{run_id},{seed},{best_acc:.4f}\n")
        if summary_csv.exists():
            mlflow.log_artifact(str(summary_csv), artifact_path="train")

        # Subir logs y el mejor modelo
        log_file = Path(paths['logs'])
        if log_file.exists():
            mlflow.log_artifact(str(log_file), artifact_path="logs")
        best_model_pth = Path(paths['best_model'])
        if best_model_pth.exists():
            mlflow.log_artifact(str(best_model_pth), artifact_path="models")

        # (Opcional) escribir un resumen de experimento
        with open(Path(paths['metrics_dir']) / "experiment_summary.csv", "w") as f:
            f.write("model_fe,run_id,seed,best_val_acc\n")
            f.write(f"{model_fe},{run_id},{seed},{best_acc:.4f}\n")
        
if __name__ == "__main__":
    main()

