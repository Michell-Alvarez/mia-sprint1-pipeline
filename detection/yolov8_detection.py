# yolov8_detection.py
import cv2
from ultralytics import YOLO
import os
from pathlib import Path
from typing import List
import shutil
from feature_engineering import is_likely_gun  # <--- importación del otro módulo

# ==============================
# Configuración principal
# ==============================
MODEL_PATH = "/home/michell-alvarez/e_modelos/EF/RoboFlow/best.pt"
INPUT_DIR = Path("/home/michell-alvarez/e_modelos/EF/RoboFlow/videos")
OUTPUT_BASE = Path("/home/michell-alvarez/e_modelos/EF/RoboFlow/clips_detectados/Violencia/Train/Violence")
ACCEPTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_CONFIDENCE = 0.25
CLIP_DURATION = 5  # segundos

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Limpiar contenido previo dentro de OUTPUT_BASE
for item in OUTPUT_BASE.iterdir():
    try:
        if item.is_file() or item.is_symlink():
            item.unlink()  # elimina archivos o enlaces simbólicos
        elif item.is_dir():
            shutil.rmtree(item)  # elimina carpetas completas
    except Exception as e:
        print(f"No se pudo eliminar {item}: {e}")
        
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
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

def process_single_video(model, video_path: Path, output_base: Path, min_conf: float, clip_seconds: int):
    print(f"\n=== Procesando: {video_path.name} ===")
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

        results = model.predict(frame, conf=min_conf, classes=[3], verbose=False)
        gun_detected = False
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            print(f"Frame {frame_idx}: YOLO detectó {len(results[0].boxes)} objetos")
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0 and is_likely_gun(box, roi):
                    gun_detected = True
                    frames_with_detection += 1
                    break

        if gun_detected:
            has_gun_in_clip = True

        current_clip_frames.append(frame)

        if len(current_clip_frames) >= frames_per_clip or frame_idx == total_frames - 1:
            if has_gun_in_clip:
                #out_path = output_base / f"{video_path.stem}_clip_{clip_counter:04d}.mp4"
                out_path = output_base / f"{video_path.stem}.mp4"
                save_clip(current_clip_frames, fps, out_path)
                print(f"Clip {clip_counter} guardado: {out_path}")
                clip_counter += 1

            current_clip_frames = []
            has_gun_in_clip = False

    cap.release()
    print(f"Finalizado: {video_path.name}, {frames_with_detection} frames con detección, {clip_counter} clips guardados")

def main():
    videos = list_videos(INPUT_DIR, ACCEPTED_EXTS)
    if not videos:
        print(f"No se encontraron videos en: {INPUT_DIR}")
        return

    print("Cargando modelo YOLOv8...")
    model = YOLO(MODEL_PATH)

    for video_path in videos:
        process_single_video(model, video_path, OUTPUT_BASE, MIN_CONFIDENCE, CLIP_DURATION)

if __name__ == "__main__":
    main()

