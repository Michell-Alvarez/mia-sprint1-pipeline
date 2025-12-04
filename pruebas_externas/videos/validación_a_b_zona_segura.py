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
CONF_THRESHOLD = 0.5
IMG_SIZE = 512

# Cómo identificar la clase "arma"
# Opción 1: por índice de clase (ej: si solo tienes 1 clase "gun", suele ser 0)
ARMA_CLASS_ID = 3

# Opción 2 (si quieres por nombre):
# Asegúrate que el nombre de la clase en tu modelo sea exactamente éste.
ARMA_CLASS_NAME = "pistol"  # cámbialo si tu clase se llama distinto, por ejemplo "arma"

# Ruta del CSV de resultados
CSV_PATH = "/home/michell-alvarez/Descargas/zona_segura/a_b_test/resultados_ab_zona_segura_videos.csv"

# Carpeta donde están los videos de zona segura
VIDEOS_DIR = "/home/michell-alvarez/Descargas/zona_segura/a_b_test"  # <-- CAMBIA ESTA RUTA
EXTENSIONES_VALIDAS = (".mp4", ".avi", ".mov", ".mkv")  # puedes agregar más


# ==========================
# FUNCIONES AUXILIARES
# ==========================

def contar_falsos_positivos(result, usar_nombre=True):
    """
    Cuenta cuántas detecciones corresponden a 'arma'
    en un frame, que en zona segura se consideran FALSOS POSITIVOS.
    """
    fp = 0
    boxes = result.boxes

    if boxes is None:
        return 0

    cls = boxes.cls.cpu().numpy()

    if usar_nombre and hasattr(result, "names"):
        names = result.names
        for c in cls:
            nombre_clase = names[int(c)]
            if nombre_clase == ARMA_CLASS_NAME:
                fp += 1
    else:
        # Por ID de clase
        for c in cls:
            if int(c) == ARMA_CLASS_ID:
                fp += 1

    return fp


def inicializar_metricas():
    return {
        "frames_procesados": 0,
        "suma_latencia_inferencia_ms": 0.0,
        "suma_latencia_total_ms": 0.0,
        "falsos_positivos": 0,
    }


def calcular_resultados(nombre_modelo, met, duracion_sec):
    fps_promedio = met["frames_procesados"] / duracion_sec if duracion_sec > 0 else 0.0
    latencia_inf_prom = (met["suma_latencia_inferencia_ms"] / met["frames_procesados"]
                         if met["frames_procesados"] > 0 else 0.0)
    latencia_total_prom = (met["suma_latencia_total_ms"] / met["frames_procesados"]
                           if met["frames_procesados"] > 0 else 0.0)
    fp_por_minuto = (met["falsos_positivos"] / (duracion_sec / 60.0)
                     if duracion_sec > 0 else 0.0)

    return {
        "modelo": nombre_modelo,
        "frames_procesados": met["frames_procesados"],
        "duracion_sec": round(duracion_sec, 2),
        "latencia_inferencia_ms": round(latencia_inf_prom, 2),
        "latencia_total_ms": round(latencia_total_prom, 2),
        "fps_promedio": round(fps_promedio, 2),
        "falsos_positivos": met["falsos_positivos"],
        "fp_por_minuto": round(fp_por_minuto, 2),
    }


def asegurar_csv_con_cabecera(ruta_csv):
    if not os.path.exists(ruta_csv):
        with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "video",                 # <-- nueva columna
                "modelo",
                "frames_procesados",
                "duracion_sec",
                "latencia_inferencia_ms",
                "latencia_total_ms",
                "fps_promedio",
                "falsos_positivos",
                "fp_por_minuto",
            ])


def append_resultado_csv(ruta_csv, resultado, nombre_video):
    with open(ruta_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            nombre_video,
            resultado["modelo"],
            resultado["frames_procesados"],
            resultado["duracion_sec"],
            resultado["latencia_inferencia_ms"],
            resultado["latencia_total_ms"],
            resultado["fps_promedio"],
            resultado["falsos_positivos"],
            resultado["fp_por_minuto"],
        ])


# ==========================
# MAIN: TEST A/B ZONA SEGURA (VIDEOS)
# ==========================

def main():
    # Cargar modelos
    print("Cargando modelos...")
    model_n = YOLO(MODEL_N_PATH)
    model_x = YOLO(MODEL_X_PATH)

    # Asegurar CSV con cabecera
    asegurar_csv_con_cabecera(CSV_PATH)

    # Verificar carpeta de videos
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

    print("=== TEST A/B - ZONA SEGURA (VIDEOS) ===")
    print(f"Carpeta: {VIDEOS_DIR}")
    print(f"Videos encontrados: {len(videos)}\n")

    # Procesar cada video de la carpeta
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
            fp_n = contar_falsos_positivos(r_n, usar_nombre=True)

            t_total_end_n = time.time()

            metricas_n["frames_procesados"] += 1
            metricas_n["falsos_positivos"] += fp_n
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
            fp_x = contar_falsos_positivos(r_x, usar_nombre=True)

            t_total_end_x = time.time()

            metricas_x["frames_procesados"] += 1
            metricas_x["falsos_positivos"] += fp_x
            metricas_x["suma_latencia_inferencia_ms"] += (t_inf_end_x - t_inf_start_x) * 1000.0
            metricas_x["suma_latencia_total_ms"] += (t_total_end_x - t_total_start_x) * 1000.0

            # (Opcional) Mostrar frame mientras corre:
            # cv2.imshow("Video - Test A/B zona segura (n & x)", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        end_time = time.time()
        duracion_sec = end_time - start_time

        cap.release()
        # cv2.destroyAllWindows()  # solo si usas imshow

        # Calcular resultados finales por modelo para este video
        resultado_n = calcular_resultados("YOLOv8n", metricas_n, duracion_sec)
        resultado_x = calcular_resultados("YOLOv8x", metricas_x, duracion_sec)

        print("\n--- RESULTADOS ZONA SEGURA PARA ESTE VIDEO ---")
        for res in [resultado_n, resultado_x]:
            print(f"\nVideo:              {video_name}")
            print(f"Modelo:             {res['modelo']}")
            print(f"Frames procesados:  {res['frames_procesados']}")
            print(f"Duración total [s]: {res['duracion_sec']}")
            print(f"Latencia inf [ms]:  {res['latencia_inferencia_ms']}")
            print(f"Latencia total [ms]:{res['latencia_total_ms']}")
            print(f"FPS promedio:       {res['fps_promedio']}")
            print(f"Falsos positivos:   {res['falsos_positivos']}")
            print(f"FP por minuto:      {res['fp_por_minuto']}")

        # Guardar resultados en CSV (una fila por modelo y video)
        append_resultado_csv(CSV_PATH, resultado_n, video_name)
        append_resultado_csv(CSV_PATH, resultado_x, video_name)

    print(f"\nProcesamiento terminado. Resultados guardados en: {CSV_PATH}")


if __name__ == "__main__":
    main()

