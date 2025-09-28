import torch
from torch.utils.data import DataLoader
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os
# Usar ruta absoluta o encontrar la ruta correcta
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'configs', 'config.yaml')
from models.cnn3d_model import Baseline3DCNN, VideoDataset

def plot_confusion_matrix(cm, classes, save_path='outputs/plots/confusion_matrix.png'):
    """Genera y guarda una matriz de confusión visual"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusión - Detección de Robo Armado')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    # Guardar imagen
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()

def evaluate_model():
    """Evalúa el modelo final con el conjunto de test"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo
    model = Baseline3DCNN().to(device)
    model.load_state_dict(torch.load(f"{config['paths']['checkpoints']}/best_model.pth"))
    model.eval()
    
    # Dataset de test
    test_dataset = VideoDataset(config['paths']['test_index'])
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Evaluación
    all_preds = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convertir a arrays numpy
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probabilities = np.array(all_probabilities)
    
    # 1. MATRIZ DE CONFUSIÓN
    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['No Robo', 'Robo Armado']  # Más descriptivo que NonViolence/Violence
    
    print("=== MATRIZ DE CONFUSIÓN ===")
    print(f"Verdaderos Negativos (TN): {cm[0, 0]}")  # No robo correcto
    print(f"Falsos Positivos (FP): {cm[0, 1]}")      # No robo predicho como robo
    print(f"Falsos Negativos (FN): {cm[1, 0]}")      # Robo predicho como no robo
    print(f"Verdaderos Positivos (TP): {cm[1, 1]}")  # Robo correcto
    print("\nMatriz completa:")
    print(cm)
    
    # 2. MÉTRICAS DETALLADAS
    report = classification_report(all_labels, all_preds, target_names=class_names)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Métricas específicas por clase
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)  # Tasa de verdaderos negativos
    sensitivity = recall = tp / (tp + fn)  # Tasa de verdaderos positivos
    
    print("\n=== MÉTRICAS DETALLADAS ===")
    print(report)
    print(f"\n=== MÉTRICAS RESUMEN ===")
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print(f"Precisión (Weighted): {precision:.4f}")
    print(f"Sensibilidad (Recall): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Especificidad: {specificity:.4f}")
    
    # 3. ANÁLISIS ADICIONAL
    print(f"\n=== ANÁLISIS POR CLASE ===")
    print(f"Robo Armado detectado correctamente: {tp}/{tp+fn} ({sensitivity:.2%})")
    print(f"No Robo identificado correctamente: {tn}/{tn+fp} ({specificity:.2%})")
    print(f"Falsas alarmas (FP): {fp} → {fp/(tp+fp):.2%} de las predicciones positivas")
    print(f"Robos no detectados (FN): {fn} → {fn/(tp+fn):.2%} de los robos reales")
    
    # 4. GRÁFICO DE MATRIZ DE CONFUSIÓN
    plot_confusion_matrix(cm, class_names)
    
    # 5. GUARDAR RESULTADOS COMPLETOS
    results = {
        'confusion_matrix': cm.tolist(),
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'sensitivity': sensitivity
        },
        'counts': {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(all_labels)
        },
        'class_distribution': {
            'actual_robos': int(tp + fn),
            'actual_no_robos': int(tn + fp)
        }
    }
    
    # Guardar resultados en CSV
    results_df = pd.DataFrame({
        'real': all_labels,
        'predicho': all_preds,
        'probabilidad_robo': all_probabilities[:, 1],
        'probabilidad_no_robo': all_probabilities[:, 0]
    })
    results_df.to_csv('outputs/metrics/detailed_results.csv', index=False)
    
    # Guardar métricas principales
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv('outputs/metrics/model_metrics.csv', index=False)
    
    print(f"\n✅ Resultados guardados en:")
    print(f"   - outputs/plots/confusion_matrix.png")
    print(f"   - outputs/metrics/detailed_results.csv")
    print(f"   - outputs/metrics/model_metrics.csv")
    
    return results

def analyze_misclassifications():
    """Análisis adicional de errores de clasificación"""
    results_df = pd.read_csv('outputs/metrics/detailed_results.csv')
    
    # Falsos positivos (alarmas falsas)
    false_positives = results_df[(results_df['real'] == 0) & (results_df['predicho'] == 1)]
    # Falsos negativos (robos no detectados)
    false_negatives = results_df[(results_df['real'] == 1) & (results_df['predicho'] == 0)]
    
    print(f"\n=== ANÁLISIS DE ERRORES ===")
    print(f"Falsos Positivos (Alarmas falsas): {len(false_positives)}")
    if len(false_positives) > 0:
        print(f"  - Probabilidad promedio de robo: {false_positives['probabilidad_robo'].mean():.3f}")
        print(f"  - Rango de probabilidad: [{false_positives['probabilidad_robo'].min():.3f}, {false_positives['probabilidad_robo'].max():.3f}]")
    
    print(f"Falsos Negativos (Robos no detectados): {len(false_negatives)}")
    if len(false_negatives) > 0:
        print(f"  - Probabilidad promedio de robo: {false_negatives['probabilidad_robo'].mean():.3f}")
        print(f"  - Rango de probabilidad: [{false_negatives['probabilidad_robo'].min():.3f}, {false_negatives['probabilidad_robo'].max():.3f}]")

if __name__ == "__main__":
    results = evaluate_model()
    analyze_misclassifications()
