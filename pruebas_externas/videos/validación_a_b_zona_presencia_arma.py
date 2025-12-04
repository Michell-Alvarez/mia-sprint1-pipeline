import time
import os
import csv
from datetime import datetime

import cv2
from ultralytics import YOLO

# ==========================
# CONFIGURACIÓN
# ==========================

# Rutas a tus modelos entrenados
MODEL_N_PATH = "/home/michell-alvarez/Descargas/best_8n.pt"
MODEL_X_PATH = "/home/michell-alvarez/Descargas/best_8x.pt"

# Umbral de confianza (el mismo para ambos en este test A/B)
CONF_THRESHOLD = 0.50
IMG_SIZE = 512

# Cómo identificar la clase "arma"
# Opción 1: por índice de clase (ej: si solo tienes 1 clase "gun", suele ser 0)
ARMA_CLASS_ID = 3

# Opción 2 (si quieres por nombre):
# Asegúrate que el nombre de la clase en tu modelo sea exactamente éste.
ARMA_CLASS_NAME = "pistol"  # cámbialo si tu clase se llama distinto, por ejemplo "arma"

USAR_NOMBRE_CLASE = True  # True = usa ARMA_CLASS_NAME, False = usa ARMA_CLASS_ID

# Ruta del CSV de resultados
CSV_PATH = "/home/michell-alvarez/Descargas/zona_peligrosa/a_b_test/resultados_ab_zona_peligrosa_videos.csv"

# Carpeta donde están los videos a evaluar
VIDEOS_DIR = "/home/michell-alvarez/Descargas/zona_peligrosa/a_b_test"  # ← CAMBIA ESTO A TU RUTA
EXTENSIONES_VALIDAS = (".mp4", ".avi", ".mov", ".mkv")  # puedes agregar más si quieres


# ==========================
# FUNCIONES AUXILIARES
# ==========================

def contar_detecciones_arma(result):
    """
    Cuenta cuántas detecciones corresponden a 'arma'
    en un frame. En este escenario, asumimos que
    cualquier detección de esa clase es un "hit" (detección de arma).
    """
    boxes = result.boxes
    if boxes is None:
        return 0

    cls = boxes.cls.cpu().numpy()
    num_dets = 0

    if USAR_NOMBRE_CLASE and hasattr(result, "names"):
        names = result.names
        for c in cls:
            nombre_clase = names[int(c)]
            if nombre_clase == ARMA_CLASS_NAME:
                num_dets += 1
    else:
        for c in cls:
            if int(c) == ARMA_CLASS_ID:
                num_dets += 1

    return num_dets


def inicializar_metricas():
    return {
        "frames_procesados": 0,
        "suma_latencia_inferencia_ms": 0.0,
        "suma_latencia_total_ms": 0.0,
        "frames_con_arma": 0,  # frames donde hubo al menos 1 detección de arma
    }


def calcular_resultados(nombre_modelo, met, duracion_sec):
    frames = met["frames_procesados"]
    fps_promedio = frames / duracion_sec if duracion_sec > 0 else 0.0

    latencia_inf_prom = (met["suma_latencia_inferencia_ms"] / frames
                         if frames > 0 else 0.0)
    latencia_total_prom = (met["suma_latencia_total_ms"] / frames
                           if frames > 0 else 0.0)


    # Escenario: este video se considera "con arma"
    tp_video = 1 if met["frames_con_arma"] > 0 else 0
    recall_video = float(tp_video)  # 1.0 si detecta al menos una vez, 0.0 si no

    return {
        "modelo": nombre_modelo,
        "frames_procesados": frames,
        "duracion_sec": round(duracion_sec, 2),
        "latencia_inferencia_ms": round(latencia_inf_prom, 2),
        "latencia_total_ms": round(latencia_total_prom, 2),
        "fps_promedio": round(fps_promedio, 2),
        "frames_con_arma": met["frames_con_arma"],
        "tp_video": tp_video,
        "recall_video": recall_video,
    }


def asegurar_csv_con_cabecera(ruta_csv):
    if not os.path.exists(ruta_csv):
        with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "video",                  
                "modelo",
                "frames_procesados",
                "duracion_sec",
                "latencia_inferencia_ms",
                "latencia_total_ms",
                "fps_promedio",
                "frames_con_arma",
                "tp_video",
                "recall_video",
            ])


def append_resultado_csv(ruta_csv, resultado, nombre_video):
    with open(ruta_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            nombre_video,  # ← para saber a qué video corresponde
            resultado["modelo"],
            resultado["frames_procesados"],
            resultado["duracion_sec"],
            resultado["latencia_inferencia_ms"],
            resultado["latencia_total_ms"],
            resultado["fps_promedio"],
            resultado["frames_con_arma"],
            resultado["tp_video"],
            resultado["recall_video"],
        ])


# ==========================
# MAIN: TEST A/B ZONA PELIGROSA CONTROLADA (VIDEOS)
# ==========================

def main():
    print("Cargando modelos...")
    model_n = YOLO(MODEL_N_PATH)
    model_x = YOLO(MODEL_X_PATH)

    # Aseguramos el CSV con cabecera
    asegurar_csv_con_cabecera(CSV_PATH)

    # Listar videos en la carpeta
    if not os.path.isdir(VIDEOS_DIR):
        print(f"La carpeta de videos no existe: {VIDEOS_DIR}")
        return

    videos = sorted([
        f for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith(EXTENSIONES_VALIDAS)
    ])

    if not videos:
        print(f"No se encontraron videos en {VIDEOS_DIR} con extensiones {EXTENSIONES_VALIDAS}")
        return

    print("=== TEST A/B - ZONA PELIGROSA CONTROLADA (VIDEOS) ===")
    print(f"Carpeta: {VIDEOS_DIR}")
    print(f"Videos encontrados: {len(videos)}\n")

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"\nProcesando video: {video_name}")
        print(f"Ruta completa: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"No se pudo abrir el video: {video_path}")
            continue

        # Métricas separadas por video y por modelo
        metricas_n = inicializar_metricas()
        metricas_x = inicializar_metricas()

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ==========================
            # MODELO N
            # ==========================
            t_total_start_n = time.time()
            t_inf_start_n = time.time()
            results_n = model_n.predict(
                frame,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False
            )
            t_inf_end_n = time.time()

            r_n = results_n[0]
            num_dets_n = contar_detecciones_arma(r_n)
            t_total_end_n = time.time()

            metricas_n["frames_procesados"] += 1
            if num_dets_n > 0:
                metricas_n["frames_con_arma"] += 1

            metricas_n["suma_latencia_inferencia_ms"] += (t_inf_end_n - t_inf_start_n) * 1000.0
            metricas_n["suma_latencia_total_ms"] += (t_total_end_n - t_total_start_n) * 1000.0

            # ==========================
            # MODELO X
            # ==========================
            t_total_start_x = time.time()
            t_inf_start_x = time.time()
            results_x = model_x.predict(
                frame,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False
            )
            t_inf_end_x = time.time()

            r_x = results_x[0]
            num_dets_x = contar_detecciones_arma(r_x)
            t_total_end_x = time.time()

            metricas_x["frames_procesados"] += 1

            if num_dets_x > 0:
                metricas_x["frames_con_arma"] += 1

            metricas_x["suma_latencia_inferencia_ms"] += (t_inf_end_x - t_inf_start_x) * 1000.0
            metricas_x["suma_latencia_total_ms"] += (t_total_end_x - t_total_start_x) * 1000.0

            # Si quisieras ver el video mientras corre (opcional):
            # cv2.imshow("Video - Test A/B (n & x)", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        end_time = time.time()
        duracion_sec = end_time - start_time

        cap.release()
        # cv2.destroyAllWindows()  # solo si usas imshow

        # Calcular resultados finales por video
        resultado_n = calcular_resultados("YOLOv8n", metricas_n, duracion_sec)
        resultado_x = calcular_resultados("YOLOv8x", metricas_x, duracion_sec)

        print("\n--- RESULTADOS A/B PARA ESTE VIDEO ---")
        for res in [resultado_n, resultado_x]:
            print(f"\nVideo:  {video_name}")
            print(f"Modelo: {res['modelo']}")
            print(f"Frames procesados:          {res['frames_procesados']}")
            print(f"Duración total [s]:         {res['duracion_sec']}")
            print(f"Latencia inf [ms]:          {res['latencia_inferencia_ms']}")
            print(f"Latencia total [ms]:        {res['latencia_total_ms']}")
            print(f"FPS promedio:               {res['fps_promedio']}")
            print(f"Frames con arma detectada:  {res['frames_con_arma']}")
            print(f"TP_video:                   {res['tp_video']}")
            print(f"Recall a nivel video:       {res['recall_video']}")

        # Guardar resultados en CSV (una fila por modelo y video)
        append_resultado_csv(CSV_PATH, resultado_n, video_name)
        append_resultado_csv(CSV_PATH, resultado_x, video_name)

    print(f"\nProcesamiento terminado. Resultados guardados en: {CSV_PATH}")


if __name__ == "__main__":
    main()

