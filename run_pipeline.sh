#!/bin/bash

# ==========================
# Validación de argumentos
# ==========================
if [ $# -lt 2 ]; then
    echo "Uso: ./run_pipeline.sh [fe_off | fe_on] [baseline | solid]"
    exit 1
fi

MODE=$1
MODEL=$2

echo "==========================================="
echo "  Ejecución del pipeline"
echo "  Modo:   $MODE"
echo "  Modelo: $MODEL"
echo "==========================================="

# ==========================
# Función del pipeline
# ==========================
run_pipeline_steps() {
    echo "Paso 3: Validando la estructura de las carpetas..."
    python data_ingest/verify_structure.py --model "$MODEL" --mode "$MODE"

    echo "Paso 4: Creando índice de videos..."
    python data_ingest/create_index.py --model "$MODEL" --mode "$MODE"

    echo "Paso 5: Dividiendo dataset..."
    python data_ingest/split_dataset.py --model "$MODEL" --mode "$MODE"

    echo "Paso 6: EDA dataset de entrenamiento..."
    python eda/eda_train.py --model "$MODEL" --mode "$MODE"

    echo "Paso 7: EDA dataset de prueba..."
    python eda/eda_test.py --model "$MODEL" --mode "$MODE"

    echo "Paso 8: EDA dataset de entrenamiento+prueba..."
    python eda/eda_train_test.py --model "$MODEL" --mode "$MODE"

    echo "Paso 9: Entrenando modelo..."
    python models/train_cnn3d_${MODEL}.py --model "$MODEL" --mode "$MODE"

    echo "Paso 10: Evaluando modelo..."
    python -m evaluation.evaluate_${MODEL} --model "$MODEL" --mode "$MODE"
    
    echo "Paso 11: Genera métricas para los modelos baseline y solid, tanto con como sin feature engineering..."
    python tools/comparar_modelos.py --base-dir outputs    
    
}

# ==========================
# Flujo principal del script
# ==========================

if [ "$MODE" == "fe_on" ]; then
    echo "Paso 1: Extracción de características: LBP, color, combinación y verificación de pistola..."
    python detection/feature_engineering.py

    echo "Paso 2: Detección y recorte de clips con YOLOv8..."
    python detection/yolov8_detection.py

    # Ejecutar el resto del pipeline
    run_pipeline_steps

elif [ "$MODE" == "fe_off" ]; then
    echo "Modo sin Feature Engineering (solo pasos 3-10)..."
    run_pipeline_steps
else
    echo "Error: modo no válido. Usa 'fe_on' o 'fe_off'."
    exit 1
fi

echo "=== Pipeline completado correctamente en modo $MODE con modelo $MODEL ==="

