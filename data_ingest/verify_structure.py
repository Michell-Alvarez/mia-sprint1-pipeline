import os
import yaml
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


def verify_folder_structure(config_path=config_path):
    """Verifica que la estructura de carpetas exista"""
    
    print(f"Buscando config en: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"❌ No se encuentra el archivo de configuración: {config_path}")
        print("💡 Creando archivo de configuración básico...")
        
        # Crear config básico si no existe
        config_dir = os.path.dirname(config_path)
        os.makedirs(config_dir, exist_ok=True)
        
        basic_config = {
            'data': {
                'train_dir': "/home/michell-alvarez/e_modelos/EF/Violencia/Train",
                'test_dir': "/home/michell-alvarez/e_modelos/EF/Violencia/Test", 
                'classes': ["NonViolence", "Violence"]
            },
            'paths': {
                'index_file': "data/video_index.csv",
                'train_index': "data/train_index.csv",
                'test_index': "data/test_index.csv"
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(basic_config, f)
        print(f"✅ Archivo de configuración creado: {config_path}")
    
    # Ahora cargar la configuración
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resto del código igual...
    train_dir = config['data']['train_dir']
    test_dir = config['data']['test_dir']
    
    print("🔍 VERIFICANDO ESTRUCTURA DE CARPETAS:")
    
    for dir_name, dir_path in [('TRAIN', train_dir), ('TEST', test_dir)]:
        if os.path.exists(dir_path):
            print(f"✅ {dir_name}: {dir_path}")
            for class_name in config['data']['classes']:
                class_path = os.path.join(dir_path, class_name)
                if os.path.exists(class_path):
                    video_count = len([f for f in os.listdir(class_path) if f.endswith('.mp4')])
                    print(f"   ✅ {class_name}: {video_count} videos")
                else:
                    print(f"   ❌ {class_name}: NO EXISTE")
        else:
            print(f"❌ {dir_name}: {dir_path} - NO EXISTE")

if __name__ == "__main__":
    verify_folder_structure()
