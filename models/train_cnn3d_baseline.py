import torch, random, numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml, logging, os
from pathlib import Path
from datetime import datetime
import argparse
from cnn3d_model_baseline import Baseline3DCNN, VideoDataset

# Crear un único parser
parser = argparse.ArgumentParser()

# Agregar ambos argumentos al mismo parser
parser.add_argument('--mode', type=str, default='fe_off', choices=['fe_off', 'fe_on'],
                    help="Modo de ejecución: con o sin feature engineering")
parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'solid'],
                    help="Modelo a ejecutar: baseline o solid")
                    
# Parsear argumentos una sola vez
args = parser.parse_args()

# Construir la ruta del archivo de configuración
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'configs', f'config_{args.model}_{args.mode}.yaml')

# ---------- 1) Reproducibilidad ----------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Para reproducibilidad en cuDNN (puede bajar algo el rendimiento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger()

def main():
    # Cargar config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Lee nombre del modelo y seed desde config
    model_name = config.get('experiment', {}).get('model', 'baseline')  # "baseline" o "solid"
    mode_fe = config.get('experiment', {}).get('mode', 'fe_off')  # "fe_off" o "fe_on"
    model_fe=model_name+'_'+mode_fe
    seed = int(config.get('experiment', {}).get('seed', 42))
    set_seed(seed)

    # Crear run_id y carpetas de salida
    run_id, run_root = make_run_dirs(model_fe, base_out=config.get('paths', {}).get('base_outputs', 'outputs'))

    # Actualizar rutas derivadas en memoria (sin tocar el YAML)
    paths = config.setdefault('paths', {})
    paths['run_root'] = str(run_root)
    paths['logs'] = str(run_root / "logs" / "training_log.txt")
    paths['checkpoints_dir'] = str(run_root / "models")
    paths['best_model'] = str(run_root / "models" / "best_model.pth")
    paths['metrics_dir'] = str(run_root / "metrics")
    #paths['plots_dir'] = str(run_root / "plots")

    # Logger
    logger = setup_logging(Path(paths['logs']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Iniciando entrenamiento en {device} | model={model_fe} | seed={seed} | run_id={run_id}")
    print(f"[INFO] model={model_fe} seed={seed} run_id={run_id} out={paths['run_root']}")

    # Datasets (split reproducible)
    full_train_dataset = VideoDataset(config['paths']['train_index'])
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    g_split = torch.Generator().manual_seed(seed)  # <- split reproducible
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=g_split)

    # DataLoaders (shuffle reproducible por época si quieres “anclarlo”)
    g_loader = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        generator=g_loader,              # <- fija el orden de barajado
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
        train_acc = correct / total

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
        val_acc = v_correct / v_total

        logger.info(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}")
        print(f"Epoch {epoch+1}: Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

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

    logger.info(f"best_val_acc,{best_accuracy:.4f}")
    # (Opcional) escribir un resumen de experimento
    with open(Path(paths['metrics_dir']) / "experiment_summary.csv", "w") as f:
        f.write("model_fe,run_id,seed,best_val_acc\n")
        f.write(f"{model_fe},{run_id},{seed},{best_accuracy:.4f}\n")

if __name__ == "__main__":
    main()

