import os
import pandas as pd
import yaml
import argparse

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

print(f"Configuración usada: {config_path}")
fe = args.fe
print(fe)
def create_video_index(config_path=config_path):
    """Crea un índice de todos los videos basado en tu estructura de carpetas existente"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data = []
    
    # Procesar carpeta de TRAIN
    #train_dir = config['data']['train_dir']
    train_dir = config['paths'][f"fe{fe}"]
    print(f"Escaneando carpeta de entrenamiento: {train_dir}")
    
    for class_name in config['data']['classes']:
        class_path = os.path.join(train_dir, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️  Advertencia: {class_path} no existe")
            continue
            
        for video_file in os.listdir(class_path):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(class_path, video_file)
                label = 1 if class_name == "Violence" else 0
                split = "train"
                
                data.append({
                    'video_path': video_path,
                    'label': label,
                    'class_name': class_name,
                    'filename': video_file,
                    'split': split
                })
                print(f"✅ Train - {class_name}: {video_file}")
    
    # Procesar carpeta de TEST
    test_dir = config['data']['test_dir']
    print(f"\nEscaneando carpeta de test: {test_dir}")
    
    for class_name in config['data']['classes']:
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️  Advertencia: {class_path} no existe")
            continue
            
        for video_file in os.listdir(class_path):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(class_path, video_file)
                label = 1 if class_name == "Violence" else 0
                split = "test"
                
                data.append({
                    'video_path': video_path,
                    'label': label,
                    'class_name': class_name,
                    'filename': video_file,
                    'split': split
                })
                print(f"✅ Test - {class_name}: {video_file}")
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Crear directorio si no existe
    # os.makedirs('../data', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Guardar índice completo
    df.to_csv(config['paths']['index_file'], index=False)
    
    # Estadísticas
    print(f"\n📊 ESTADÍSTICAS DEL DATASET:")
    print(f"Total de videos: {len(df)}")
    print(f"Train - Violence: {len(df[(df['split']=='train') & (df['label']==1)])}")
    print(f"Train - NonViolence: {len(df[(df['split']=='train') & (df['label']==0)])}")
    print(f"Test - Violence: {len(df[(df['split']=='test') & (df['label']==1)])}")
    print(f"Test - NonViolence: {len(df[(df['split']=='test') & (df['label']==0)])}")
    
    print(f"\n💾 Índice guardado en: {config['paths']['index_file']}")
    return df

if __name__ == "__main__":
    create_video_index()
