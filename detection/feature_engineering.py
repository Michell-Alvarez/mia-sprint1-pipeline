# feature_engineering.py
import cv2
import numpy as np

def extract_lbp_features(roi):
    """Versión alternativa de LBP usando implementación manual"""
    if roi.size == 0:
        return {'lbp_uniformity': 0, 'lbp_contrast': 0}
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    lbp_image = np.zeros_like(gray)

    for i in range(1, height-1):
        for j in range(1, width-1):
            center = gray[i, j]
            code = 0
            code |= (gray[i-1, j-1] > center) << 7
            code |= (gray[i-1, j] > center) << 6
            code |= (gray[i-1, j+1] > center) << 5
            code |= (gray[i, j+1] > center) << 4
            code |= (gray[i+1, j+1] > center) << 3
            code |= (gray[i+1, j] > center) << 2
            code |= (gray[i+1, j-1] > center) << 1
            code |= (gray[i, j-1] > center) << 0
            lbp_image[i, j] = code
    
    hist_lbp = cv2.calcHist([lbp_image], [0], None, [256], [0, 256])
    lbp_uniformity = np.sum(hist_lbp[:59]) / max(1e-9, np.sum(hist_lbp))
    return {
        'lbp_uniformity': float(lbp_uniformity),
        'lbp_contrast': float(np.std(hist_lbp))
    }

def extract_gun_color_features(roi):
    """Extrae características de color de la región de interés"""
    if roi.size == 0:
        return {'metallic_color_ratio': 0, 'black_color_ratio': 0, 'color_uniformity': 100}
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    total = float(np.sum(hist_h))
    if total <= 0:
        return {'metallic_color_ratio': 0, 'black_color_ratio': 0, 'color_uniformity': 100}
    
    metallic_ratio = float(np.sum(hist_h[8:12]) / total)
    black_ratio = float(np.sum(hist_h[0:2]) / total)
    uniformity = float(np.std(hist_h))
    return {
        'metallic_color_ratio': metallic_ratio,
        'black_color_ratio': black_ratio,
        'color_uniformity': uniformity
    }

def extract_combined_features(roi):
    """Combina características de color y textura"""
    color_features = extract_gun_color_features(roi)
    texture_features = extract_lbp_features(roi)
    return {**color_features, **texture_features}

def is_likely_gun(yolo_detection, frame_region):
    """Verifica si la detección de YOLO tiene características de pistola"""
    if frame_region.size == 0:
        return False
        
    features = extract_combined_features(frame_region)
    
    print(f"Metálico: {features['metallic_color_ratio']:.3f}, "
          f"Negro: {features['black_color_ratio']:.3f}, "
          f"Uniformidad_color: {features['color_uniformity']:.3f}, "
          f"LBP_Uniformidad: {features['lbp_uniformity']:.3f}, "
          f"LBP_Contraste: {features['lbp_contrast']:.3f}")
    
    color_ok = (features['metallic_color_ratio'] > 0.1 or features['black_color_ratio'] > 0.3)
    texture_ok = (features['lbp_uniformity'] > 0.3)
    
    if not texture_ok:
        print(f"   Textura rechazada: LBP_uniformity = {features['lbp_uniformity']:.3f} < 0.3")
    
    if color_ok and texture_ok:
        print("DETECCIÓN CONFIRMADA (color + textura)")
        return True
    else:
        print(f"Detección rechazada - Color: {color_ok}, Textura: {texture_ok}")
        return False

