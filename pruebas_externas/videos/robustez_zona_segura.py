import cv2
from ultralytics import YOLO
import time
import os
import pandas as pd

# ==============================
# CONFIGURACIÓN
# ==============================
MODEL_PATH = "/home/michell-alvarez/Descargas/best_8x.pt"
IMG_SIZE = 512
CONF_THRESHOLD = 0.5

VIDEO_FOLDER = "/home/michell-alvarez/Descargas/zona_segura/hold-out"

# Extensiones válidas de video
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")

# ==============================
# CARGAR MODELO
# ==============================
model = YOLO(MODEL_PATH)

# Buscar índice de clase "pistol"
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
videos_sin_arma = []
for file in os.listdir(VIDEO_FOLDER):
    if file.lower().endswith(VIDEO_EXT):
        videos_sin_arma.append(os.path.join(VIDEO_FOLDER, file))

videos_sin_arma.sort()  # ordenar alfabéticamente

print(f"Total de videos encontrados: {len(videos_sin_arma)}")


resultados_totales = []

for video_path in videos_sin_arma:
    if not os.path.exists(video_path):
        print(f" No existe el video: {video_path}")
        continue

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" No se pudo abrir el video: {video_path}")
        continue

    print(f"\n=== Evaluando video (zona segura): {video_path} ===")

    n_frames = 0
    acum_infer_ms = 0.0
    acum_total_ms = 0.0
    false_positives = 0


    while True:
        t_total_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break  # fin del video

        t_infer_start = time.time()
        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            classes=[IDX_PISTOL],
            verbose=False
        )
        t_infer_end = time.time()
        infer_ms = (t_infer_end - t_infer_start) * 1000.0

        detections = results[0].boxes
        pistol_detected = len(detections) > 0

        # En zona segura, cualquier detección = falso positivo
        if pistol_detected:
            false_positives += 1

        t_total_end = time.time()
        total_ms = (t_total_end - t_total_start) * 1000.0

        n_frames += 1
        acum_infer_ms += infer_ms
        acum_total_ms += total_ms

    cap.release()
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    if fps_video and fps_video > 0:
        duration_sec = n_frames / fps_video
    else:
        fps_video = 30.0      # fallback típico
        duration_sec = n_frames / fps_video

    if n_frames > 0:
        avg_infer_ms = acum_infer_ms / n_frames
        avg_total_ms = acum_total_ms / n_frames
        avg_fps = 1000.0 / avg_total_ms
        fp_per_min = false_positives / (duration_sec / 60.0) if duration_sec > 0 else 0.0

        print(f"Frames procesados: {n_frames}")
        print(f"Duración total: {duration_sec:.2f} s")
        print(f"Latencia promedio de inferencia: {avg_infer_ms:.2f} ms")
        print(f"Latencia promedio total: {avg_total_ms:.2f} ms")
        print(f"FPS promedio: {avg_fps:.2f}")
        print(f"Falsos positivos totales: {false_positives}")
        print(f"Falsos positivos por minuto: {fp_per_min:.2f}")

        # Guardar en lista (una fila por video)
        resultados_totales.append({
            "video": os.path.basename(video_path),  # <-- pon aquí el nombre del video
            "frames_procesados": n_frames,
            "duracion_sec": round(duration_sec, 2),
            "latencia_inferencia_ms": round(avg_infer_ms, 2),
            "latencia_total_ms": round(avg_total_ms, 2),
            "fps_promedio": round(avg_fps, 2),
            "falsos_positivos": false_positives,
            "fp_por_minuto": round(fp_per_min, 2)
        })

df = pd.DataFrame(resultados_totales)
df.to_excel("/home/michell-alvarez/Descargas/zona_segura/hold-out/robustez_zona_segura.xlsx", index=False)
print("Excel generado correctamente.")
