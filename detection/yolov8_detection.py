# yolov8_detection.py
import cv2
from ultralytics import YOLO
import os
from pathlib import Path
from typing import List
import shutil
import argparse
from feature_engineering import is_likely_gun  # usa firma: is_likely_gun(roi, fe=1)

# ==============================
# Configuración principal
# ==============================
MODEL_PATH = "/home/michell-alvarez/e_modelos/EF/RoboFlow/best.pt"
INPUT_DIR = Path("/home/michell-alvarez/e_modelos/EF/RoboFlow/videos")
ACCEPTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_CONFIDENCE = 0.25
CLIP_DURATION = 5  # segundos

# Mapeo de rutas por modo
OUTPUT_BASES = {
    1: Path("/home/michell-alvarez/e_modelos/EF/RoboFlow/clips_detectados/combined_features/Violencia/Train/Violence"),
    2: Path("/home/michell-alvarez/e_modelos/EF/RoboFlow/clips_detectados/color_features/Violencia/Train/Violence"),
    3: Path("/home/michell-alvarez/e_modelos/EF/RoboFlow/clips_detectados/lbp_features/Violencia/Train/Violence"),
}

# ==============================
# Utilidades multi-video
# ==============================
def list_videos(input_dir: Path, exts: set) -> List[Path]:
    vids = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    vids.sort()
    return vids

def ensure_fps(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps if fps and fps > 0 else 30.0)

def save_clip(frames, fps, out_path: Path):
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

def clean_output_dir(output_base: Path):
    """Limpia solo la carpeta del modo actual."""
    output_base.mkdir(parents=True, exist_ok=True)
    for item in output_base.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"No se pudo eliminar {item}: {e}")

def process_single_video(model, video_path: Path, output_base: Path, min_conf: float, clip_seconds: int, fe: int):
    print(f"\n=== Procesando: {video_path.name} (modo {fe}) ===")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ADVERTENCIA] No se pudo abrir: {video_path}")
        return

    fps = ensure_fps(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = max(1, int(clip_seconds * fps))
    print(f"Video: {total_frames} frames, {fps:.2f} fps")

    current_clip_frames, clip_counter = [], 0
    has_gun_in_clip, frames_with_detection = False, 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Detección YOLO (clase 3: pistola según tu set-up)
        results = model.predict(frame, conf=min_conf, classes=[3], verbose=False)
        gun_detected = False

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Si hay múltiples boxes, con que uno cumpla is_likely_gun, marcamos el frame
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                # Clip a los límites de la imagen:
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                roi = frame[y1:y2, x1:x2]

                # Llamada con modo seleccionado
                if roi.size > 0 and is_likely_gun(roi, fe=fe):
                    gun_detected = True
                    frames_with_detection += 1
                    break  # no necesitamos más cajas para este frame

        if gun_detected:
            has_gun_in_clip = True

        current_clip_frames.append(frame)

        # Cierre de clip por duración o fin de video
        if len(current_clip_frames) >= frames_per_clip or frame_idx == total_frames - 1:
            if has_gun_in_clip:
                # Guarda con sufijo de clip para evitar sobrescribir
                #out_path = output_base / f"{video_path.stem}_clip_{clip_counter:04d}.mp4"
                out_path = output_base / f"{video_path.stem}.mp4"
                save_clip(current_clip_frames, fps, out_path)
                print(f"Clip {clip_counter} guardado: {out_path}")
                clip_counter += 1

            # Reset para el siguiente clip
            current_clip_frames = []
            has_gun_in_clip = False

    cap.release()
    print(f"Finalizado: {video_path.name}, {frames_with_detection} frames con detección, {clip_counter} clips guardados")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fe", type=int, default=1, choices=[1, 2, 3],
                        help="1: combined_features | 2: color_features | 3: lbp_features")
    parser.add_argument("--clean", action="store_true",
                        help="Si se pasa, limpia la carpeta de salida del modo actual antes de procesar")
    args = parser.parse_args()

    fe = args.fe
    output_base = OUTPUT_BASES[fe]

    if args.clean:
        print(f"Limpieza de carpeta de salida para modo {fe}: {output_base}")
        clean_output_dir(output_base)
    else:
        output_base.mkdir(parents=True, exist_ok=True)

    videos = list_videos(INPUT_DIR, ACCEPTED_EXTS)
    if not videos:
        print(f"No se encontraron videos en: {INPUT_DIR}")
        return

    print("Cargando modelo YOLOv8...")
    model = YOLO(MODEL_PATH)

    for video_path in videos:
        process_single_video(model, video_path, output_base, MIN_CONFIDENCE, CLIP_DURATION, fe)

if __name__ == "__main__":
    main()

