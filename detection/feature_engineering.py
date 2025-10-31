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


def is_likely_gun(frame_region, fe=1):
    """
    Verifica si la detección tiene características de pistola según el modo:
    fe=1 → color + textura (combinado)
    fe=2 → solo color
    fe=3 → solo textura
    """
    if frame_region.size == 0:
        return False

    # Selección de características según modo
    if fe == 1:
        features = extract_combined_features(frame_region)
        method = "COMBINADO"
    elif fe == 2:
        features = extract_gun_color_features(frame_region)
        method = "SOLO COLOR"
    elif fe == 3:
        features = extract_lbp_features(frame_region)
        method = "SOLO TEXTURA"
    else:
        raise ValueError("Modo inválido: solo se acepta 1, 2 o 3")

    print(f"\n[Modo {fe} - {method}]")

    # Mostrar solo las métricas relevantes en cada modo
    for k, v in features.items():
        print(f"{k}: {v:.3f}")

    # Reglas de decisión según modo
    if fe == 1:
        color_ok = (features['metallic_color_ratio'] > 0.1 or features['black_color_ratio'] > 0.3)
        texture_ok = (features['lbp_uniformity'] > 0.3)
        return color_ok and texture_ok

    if fe == 2:
        return (features['metallic_color_ratio'] > 0.1 or features['black_color_ratio'] > 0.35)

    if fe == 3:
        return (features['lbp_uniformity'] > 0.3)

