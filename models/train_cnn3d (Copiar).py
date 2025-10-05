import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import logging
import os

from cnn3d_model import Baseline3DCNN, VideoDataset

# Usar ruta absoluta o encontrar la ruta correcta
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'configs', 'config.yaml')

def setup_logging(config):
    """Configura el sistema de logging"""
    log_dir = os.path.dirname(config['paths']['logs'])
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=config['paths']['logs'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger() 
    
def train_epoch(model, loader, criterion, optimizer, device):
    """Entrena una época completa"""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    """Evalúa el modelo"""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), correct / total

def main():
    # Cargar configuración
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging(config)
    
    logger.info(f"Iniciando entrenamiento en {device}")
    
    # Datasets
    #train_dataset = VideoDataset(config['paths']['train_index'])
    #val_dataset = VideoDataset(config['paths']['test_index'])
    
    full_train_dataset = VideoDataset(config['paths']['train_index'])
    
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Modelo
    model = Baseline3DCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Entrenamiento
    best_accuracy = 0.0
    patience_counter = 0
    
    logger.info("Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc")
    
    for epoch in range(config['training']['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Logging
        logger.info(f"{epoch+1}\t{train_loss:.4f}\t{train_acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}")
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Checkpoint
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            patience_counter = 0
            log_dir = os.path.dirname(config['paths']['checkpoints'])
            os.makedirs(log_dir, exist_ok=True)
            model_save_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= config['training']['early_stop_patience']:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Mejor accuracy de validación: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
