import yaml
import os
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

'''
# Usar ruta absoluta o encontrar la ruta correcta
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'configs', 'config.yaml')

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='fe_off', choices=['fe_off', 'fe_on'])
args = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'configs', f'config_baseline_{args.mode}.yaml')
'''

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


# Ahora cargar la configuración
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

#dataset_root = config['data']['dataset_root']
train_dir = os.path.join(config['data']['train_dir'])
test_dir = os.path.join(config['data']['test_dir'])

classes = config['data']['classes']

# Lista para guardar la información combinada
data_info = []

# Recorremos ambos conjuntos (train y test)
for split_name, base_dir in [('train', train_dir), ('test', test_dir)]:
    classes = os.listdir(base_dir)
    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for vid in os.listdir(class_dir):
            if not vid.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue
            video_path = os.path.join(class_dir, vid)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[Error] No se pudo abrir: {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            data_info.append([
                split_name, cls, video_path, vid,
                frame_count, fps, width, height, duration
            ])

            cap.release()

# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

# Crear DataFrame final unificado
df_all = pd.DataFrame(data_info, columns=[
    "split", "class", "video_path", "filename",
    "frames", "fps", "width", "height", "duration"
])

# Mostrar resumen
print("########################################################################################################################")
print("Resumen general de videos (train + test):")
print(df_all.head())
print(f"Total videos: {len(df_all)} (train={len(df_all[df_all.split=='train'])}, test={len(df_all[df_all.split=='test'])})")


# 🔹 Redondear la columna 'duration' a 2 decimales
df_all["duration"] = df_all["duration"].round(2)

# Guardar a CSV
df_all.to_csv("eda/metadata/train_test_videos.csv", index=False)


print("########################################################################################################################")
# ---------- 2. Distribución de clases ----------
plt.figure(figsize=(6,4))
sns.countplot(data=df_all, x="class")
plt.title("Distribución de clases")
# Guardar en una ruta específica
plt.savefig("eda/plots/distribucion_clases_consolidado.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()  # <- Cierra la figura, así no se renderiza en pantalla

print("########################################################################################################################")
# ---------- 3. Propiedades técnicas ----------
print("Duración promedio (segundos):", df_all["duration"].mean())
print("Duración mínima:", df_all["duration"].min())
print("Duración máxima:", df_all["duration"].max())

print("Resoluciones más comunes:")
print(df_all[["width","height"]].value_counts().head())
print("FPS promedio:", df_all["fps"].mean())

print("########################################################################################################################")
# ---------- 4. Histogramas ----------
plt.figure(figsize=(8,4))
sns.histplot(df_all["duration"], bins=20, kde=True)
plt.title("Distribución de duración de los videos (s)")
plt.xlabel("Duración (segundos)")
# Guardar en una ruta específica
plt.savefig("eda/plots/distribucion_de_duracion_videos_consolidado.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()  # <- Cierra la figura, así no se renderiza en pantalla

plt.figure(figsize=(8,4))
sns.histplot(df_all["fps"], bins=20, kde=True)
plt.title("Distribución de FPS")
plt.xlabel("FPS")
# Guardar en una ruta específica
plt.savefig("eda/plots/distribucion_de_fps_consolidado.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()  # <- Cierra la figura, así no se renderiza en pantalla


