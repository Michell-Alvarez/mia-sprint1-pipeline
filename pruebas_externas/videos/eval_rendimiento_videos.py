import cv2
from ultralytics import YOLO
import time
import pandas as pd
import os

# ======================
# 1. Cargar modelo YOLO
# ======================
model = YOLO("/home/michell-alvarez/Descargas/best_8x.pt")

print("Clases del modelo:", model.names)

# Buscar índice de clase "pistol"
IDX_PISTOL = None
for k, v in model.names.items():
    if v.lower() in ["pistol", "gun", "arma", "revolver"]:
        IDX_PISTOL = k
        break

if IDX_PISTOL is None:
    raise ValueError("⚠️ No se encontró la clase pistola en el modelo.")

print(f"Índice de pistola: {IDX_PISTOL}")

# ======================
# 2. Abrir VIDEO DESDE ARCHIVO
# ======================
VIDEO_PATH = "/home/michell-alvarez/Descargas/video_145.mkv"  
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ No se pudo abrir el video: {VIDEO_PATH}")
    raise SystemExit

cv2.namedWindow("YOLO Pistola - Alerta", cv2.WINDOW_NORMAL)

CONF_THRESHOLD = 0.50
IMG_SIZE = 416  # para que coincida con el Excel 640, 512 o 416


# Estadísticas
n_frames = 0
acum_infer_ms = 0.0
acum_total_ms = 0.0

# Timer
start_time = time.time()
MAX_SECONDS = 40  # ⏱️ límite de 40 segundos de ejecución

while True:
    t_total_start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("✅ Fin del video o no se pudo leer más frames.")
        break

    # ---- Inferencia ----
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

    annotated = results[0].plot()
    detections = results[0].boxes
    pistol_detected = len(detections) > 0

    # ---- Banner de alerta ----
    if pistol_detected:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 60), (0, 0, 255), -1)
        cv2.putText(annotated, "⚠️ PISTOLA DETECTADA", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        thickness = 8
        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1] - 1, annotated.shape[0] - 1),
            (0, 0, 255),
            thickness
        )
    else:
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 60), (0, 255, 0), -1)
        cv2.putText(annotated, "🟢 ZONA SEGURA", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        thickness = 6
        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1] - 1, annotated.shape[0] - 1),
            (0, 255, 0),
            thickness
        )

    # ---- Métricas ----
    t_total_end = time.time()
    total_ms = (t_total_end - t_total_start) * 1000.0
    fps_inst = 1000.0 / total_ms if total_ms > 0 else 0.0

    n_frames += 1
    acum_infer_ms += infer_ms
    acum_total_ms += total_ms

    # Tiempo transcurrido
    elapsed_sec = time.time() - start_time
    elapsed_txt = f"Tiempo: {elapsed_sec:.1f}s"

    # ⏱️🔥 CORTE AUTOMÁTICO A LOS 40 S
    if elapsed_sec >= MAX_SECONDS:
        print("⏹️ Tiempo límite alcanzado (40 segundos). Finalizando...")
        break

    # Posiciones
    h, w = annotated.shape[:2]
    y1 = h - 75
    y2 = h - 50
    y3 = h - 25

    cv2.rectangle(annotated, (0, h - 85), (w, h), (0, 0, 0), -1)

    cv2.putText(annotated, elapsed_txt, (10, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Inferencia: {infer_ms:.1f} ms", (10, y2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Total: {total_ms:.1f} ms  |  FPS inst.: {fps_inst:.1f}", (10, y3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("YOLO Pistola - Alerta", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("⏹️ Corte manual por teclado (q).")
        break

# ---- Cierre ----
cap.release()
cv2.destroyAllWindows()

# ---- Resumen final ----
end_time = time.time()
duration_sec = end_time - start_time

if n_frames > 0:
    lat_infer_prom = acum_infer_ms / n_frames
    lat_total_prom = acum_total_ms / n_frames
    fps_prom = 1000.0 / (acum_total_ms / n_frames)

    print(f"Frames procesados: {n_frames}")
    print(f"Duración total: {duration_sec:.2f} segundos")
    print(f"Latencia promedio de inferencia: {lat_infer_prom:.2f} ms")
    print(f"Latencia promedio total: {lat_total_prom:.2f} ms")
    print(f"FPS promedio: {fps_prom:.2f}")
else:
    print("⚠️ No se procesaron frames, no se guardarán métricas.")
    exit()

# ============================
#  GUARDAR RESULTADOS EN EXCEL
# ============================

excel_path = "/home/michell-alvarez/Descargas/resultados_detec_videos_baseline.xlsx"

video_name = os.path.basename(VIDEO_PATH)

data = {
    "video_name": [video_name],
    "frames": [n_frames],
    "duracion_sec": [duration_sec],
    "latencia_inferencia_ms": [lat_infer_prom],
    "latencia_total_ms": [lat_total_prom],
    "fps_promedio": [fps_prom],
    "imgsz": [IMG_SIZE]
}

df = pd.DataFrame(data)

if not os.path.exists(excel_path):
    df.to_excel(excel_path, index=False)
    print(f"📄 Archivo creado: {excel_path}")
else:
    df_existente = pd.read_excel(excel_path)
    df_final = pd.concat([df_existente, df], ignore_index=True)
    df_final.to_excel(excel_path, index=False)
    print(f"📄 Resultados agregados a: {excel_path}")
