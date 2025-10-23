import pandas as pd
import yaml
import os
import argparse

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

print(f"Configuración usada: {config_path}")

def create_split_files(config_path=config_path):
    """Crea archivos separados para train y test basados en el índice"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Cargar índice completo
    try:
        df = pd.read_csv(config['paths']['index_file'])
        print(f"📁 Índice cargado: {len(df)} videos")
    except FileNotFoundError:
        print("❌ Error: Primero ejecuta create_index.py")
        return
    
    # Separar en train y test
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    # Guardar splits
    train_df.to_csv(config['paths']['train_index'], index=False)
    test_df.to_csv(config['paths']['test_index'], index=False)
    
    print(f"\n📂 ARCHIVOS CREADOS:")
    print(f"Train: {config['paths']['train_index']} ({len(train_df)} videos)")
    print(f"Test: {config['paths']['test_index']} ({len(test_df)} videos)")
    
    # Verificar que los archivos de video existen
    print(f"\n🔍 VERIFICANDO ARCHIVOS:")
    
    missing_files = []
    for idx, row in df.iterrows():
        if not os.path.exists(row['video_path']):
            missing_files.append(row['video_path'])
    
    if missing_files:
        print(f"⚠️  Archivos faltantes: {len(missing_files)}")
        for missing in missing_files[:3]:  # Mostrar solo los primeros 3
            print(f"   - {missing}")
        if len(missing_files) > 3:
            print(f"   ... y {len(missing_files) - 3} más")
    else:
        print("✅ Todos los archivos de video existen")

if __name__ == "__main__":
    create_split_files()

