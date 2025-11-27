import cv2
from ultralytics import YOLO
import pandas as pd
import os

MODEL_PATH = "/home/michell-alvarez/Descargas/best_8n.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.5

model = YOLO(MODEL_PATH)

# Buscar índice de "pistol"
IDX_PISTOL = None
for k, v in model.names.items():
    if v.lower() in ["pistol", "gun", "arma", "revolver"]:
        IDX_PISTOL = k
        break

if IDX_PISTOL is None:
    raise ValueError("No se encontró la clase pistola en el modelo.")

# Lista de videos con arma (zona peligrosa)
videos_con_arma = [
    "/home/michell-alvarez/Descargas/zona_peligrosa/1_Distancia_Arma_Pequena.mp4",
    "/home/michell-alvarez/Descargas/zona_peligrosa/2_Arma_Altura_Media.mp4",
    "/home/michell-alvarez/Descargas/zona_peligrosa/3_Movimiento_Rapido_Arma.mp4",
    "/home/michell-alvarez/Descargas/zona_peligrosa/4_Robo_Mano_Armada_Distancia_Arma_Lejana.mp4",
    "/home/michell-alvarez/Descargas/zona_peligrosa/5_Robo_Mano_Armada_Movimiento_Camina.mp4",
    "/home/michell-alvarez/Descargas/zona_peligrosa/6_Robo_Mano_Armada_Oclusion_Cuerpo.mp4",
    "/home/michell-alvarez/Descargas/zona_peligrosa/7_Movimiento_Rapido_Arma.mp4",
    "/home/michell-alvarez/Descargas/zona_peligrosa/8_Noche.mp4",
    "/home/michell-alvarez/Descargas/zona_peligrosa/9_Oclusion_Parcial.mp4",
    "/home/michell-alvarez/Descargas/zona_peligrosa/10_Oclusion_Parcial_Brazo.mp4",
]

total_videos = 0
videos_tp = 0  # videos donde el modelo detectó al menos una vez
videos_fn = 0  # videos donde nunca detectó

resultados_totales = []   # aquí guardamos una fila por video

for video_path in videos_con_arma:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" No se pudo abrir el video: {video_path}")
        continue

    print(f"\n=== Evaluando video (zona peligrosa): {video_path} ===")

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

        detections = results[0].boxes
        pistol_detected = len(detections) > 0

        if pistol_detected:
            detecto_en_alguno = True
            # Si solo te interesa saber si detecta al menos una vez, puedes cortar aquí:
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

    # Guardamos resultados a nivel de video
    resultados_totales.append({
        "video": os.path.basename(video_path),
        "frames_procesados": n_frames,
        "TP_video": tp_video,
        "FN_video": fn_video
    })

print("\n====== Resultados a nivel de video ======")
print(f"Total de videos con arma:       {total_videos}")
print(f"Videos detectados (TP):         {videos_tp}")
print(f"Videos NO detectados (FN):      {videos_fn}")

if total_videos > 0:
    recall_video = videos_tp / total_videos
    print(f"Recall aproximado a nivel video: {recall_video:.3f}")

# ==============================
# Guardar resultados en Excel
# ==============================
if resultados_totales:
    df = pd.DataFrame(resultados_totales)
    output_path = "/home/michell-alvarez/Descargas/zona_peligrosa/robustez_zona_con_presencia_arma.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\n Resultados guardados en: {output_path}")
else:
    print("\n No se generaron resultados para guardar en Excel.")
