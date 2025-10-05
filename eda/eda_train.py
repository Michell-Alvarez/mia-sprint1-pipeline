import yaml
import os
import cv2
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Usar ruta absoluta o encontrar la ruta correcta
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'configs', 'config.yaml')

# Ahora cargar la configuración
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

#dataset_root = config['data']['dataset_root']
train_dir = os.path.join(config['data']['train_dir'])
classes = config['data']['classes']

#print("Dataset root:", dataset_root)
print("Train directory:", train_dir)
print("Classes:", classes)


# ---------- 1. Información general del dataset ----------
# Estructura: dataset_root/class_name/*.mp4

classes = os.listdir(train_dir)
data_info_train = []

for cls in classes:
    class_dir = os.path.join(train_dir, cls)
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
        data_info_train.append([cls, video_path, frame_count, fps, width, height, duration])

        cap.release()

       
# Configurar pandas para mostrar todas las columnas sin truncamiento
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.max_colwidth', None)  # Mostrar contenido completo de todas las columnas
pd.set_option('display.width', None)  # Sin límite de ancho de pantalla
pd.set_option('display.max_rows', None)  # Mostrar todas las filas (cuidado con datasets muy grandes)

df_train = pd.DataFrame(data_info_train, columns=["class", "video_path", "frames", "fps", "width", "height", "duration"])
print("########################################################################################################################")
print("Resumen general videos de entrenamiento:")
print(df_train.head())  # Usar df en lugar de df.head() para mostrar todo
print("Número total de videos:", len(df_train))


# 🔹 Redondear la columna 'duration' a 2 decimales
df_train["duration"] = df_train["duration"].round(2)

# Guardar a CSV
df_train.to_csv("eda/metadata/train_videos.csv", index=False)


print("########################################################################################################################")
# ---------- 2. Distribución de clases ----------
plt.figure(figsize=(6,4))
sns.countplot(data=df_train, x="class")
plt.title("Distribución de clases")
# Guardar en una ruta específica
plt.savefig("eda/plots/distribucion_clases_train.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()  # <- Cierra la figura, así no se renderiza en pantalla

print("########################################################################################################################")
# ---------- 3. Propiedades técnicas ----------
print("Duración promedio (segundos):", df_train["duration"].mean())
print("Duración mínima:", df_train["duration"].min())
print("Duración máxima:", df_train["duration"].max())

print("Resoluciones más comunes:")
print(df_train[["width","height"]].value_counts().head())
print("FPS promedio:", df_train["fps"].mean())

print("########################################################################################################################")
# ---------- 4. Histogramas ----------
plt.figure(figsize=(8,4))
sns.histplot(df_train["duration"], bins=20, kde=True)
plt.title("Distribución de duración de los videos (s)")
plt.xlabel("Duración (segundos)")
# Guardar en una ruta específica
plt.savefig("eda/plots/distribucion_de_duracion_videos_train.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()  # <- Cierra la figura, así no se renderiza en pantalla

plt.figure(figsize=(8,4))
sns.histplot(df_train["fps"], bins=20, kde=True)
plt.title("Distribución de FPS")
plt.xlabel("FPS")
# Guardar en una ruta específica
plt.savefig("eda/plots/distribucion_de_fps_train.png", dpi=300, bbox_inches="tight")
#plt.show()
plt.close()  # <- Cierra la figura, así no se renderiza en pantalla


