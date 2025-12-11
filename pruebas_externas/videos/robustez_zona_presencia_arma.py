import cv2
from ultralytics import YOLO
import pandas as pd
import os

# ==============================
# CONFIGURACIÓN
# ==============================
MODEL_PATH = "/home/michell-alvarez/Descargas/best_8x.pt"
IMG_SIZE = 512
CONF_THRESHOLD = 0.5
VIDEO_FOLDER = "/home/michell-alvarez/Descargas/zona_peligrosa/hold-out"

# Extensiones válidas de video
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")

# ==============================
# CARGAR MODELO
# ==============================
model = YOLO(MODEL_PATH)

# Buscar índice de "pistol"
IDX_PISTOL = None
for k, v in model.names.items():
    if v.lower() in ["pistol", "gun", "arma", "revolver"]:
        IDX_PISTOL = k
        break

if IDX_PISTOL is None:
    raise ValueError("No se encontró la clase pistola en el modelo.")

# ==============================
# LEER TODOS LOS VIDEOS DE LA CARPETA
# ==============================
videos_con_arma = []
for file in os.listdir(VIDEO_FOLDER):
    if file.lower().endswith(VIDEO_EXT):
        videos_con_arma.append(os.path.join(VIDEO_FOLDER, file))

videos_con_arma.sort()  # ordenar alfabéticamente

print(f"Total de videos encontrados: {len(videos_con_arma)}")

# ==============================
# PROCESAR CADA VIDEO
# ==============================
total_videos = 0
videos_tp = 0
videos_fn = 0
resultados_totales = []

for video_path in videos_con_arma:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" No se pudo abrir el video: {video_path}")
        continue

    print(f"\n=== Evaluando video (zona peligrosa): {os.path.basename(video_path)} ===")

    total_videos += 1
    detecto_en_alguno = False
    n_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        n_frames += 1

        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            classes=[IDX_PISTOL],
            verbose=False
        )

        pistol_detected = len(results[0].boxes) > 0

        if pistol_detected:
            detecto_en_alguno = True
            # si solo necesitas saber si detecta una vez, podrías cortar aquí:
            # break

    cap.release()

    if detecto_en_alguno:
        videos_tp += 1
        tp_video = 1
        fn_video = 0
        print(" Detectó al menos una vez (TP)")
    else:
        videos_fn += 1
        tp_video = 0
        fn_video = 1
        print(" Nunca detectó (FN)")

    resultados_totales.append({
        "video": os.path.basename(video_path),
        "frames_procesados": n_frames,
        "TP_video": tp_video,
        "FN_video": fn_video
    })

# ==============================
# RESULTADOS FINALES
# ==============================
print("\n====== Resultados a nivel de video ======")
print(f"Total de videos:                 {total_videos}")
print(f"Videos detectados (TP):          {videos_tp}")
print(f"Videos NO detectados (FN):       {videos_fn}")

if total_videos > 0:
    recall_video = videos_tp / total_videos
    print(f"Recall a nivel video:            {recall_video:.3f}")

# ==============================
# GUARDAR EN EXCEL
# ==============================
if resultados_totales:
    df = pd.DataFrame(resultados_totales)
    output_path = os.path.join(VIDEO_FOLDER, "robustez_zona_con_presencia_arma.xlsx")
    df.to_excel(output_path, index=False)
    print(f"\n Resultados guardados en: {output_path}")
else:
    print("\n No se generaron resultados para guardar en Excel.")

