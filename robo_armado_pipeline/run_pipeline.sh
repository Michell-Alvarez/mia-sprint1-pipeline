#!/bin/bash

echo "=== Pipeline de Detección de Robo Armado ==="

# 1. Validar estructura de las carpetas
echo "Paso 1: Validando la estructura de las carpetas..."
python data_ingest/verify_structure.py

# 2. Crear índice de videos
echo "Paso 2: Creando índice de videos..."
python data_ingest/create_index.py

# 3. Dividir dataset
echo "Paso 3: Dividiendo dataset..."
python data_ingest/split_dataset.py

# 4. 
echo "Paso 4: EDA dataset de entrenamiento..."
python eda/eda_train.py

# 5. 
echo "Paso 5: EDA dataset de prueba..."
python eda/eda_test.py

# 6. Entrenar modelo
echo "Paso 6: Entrenando modelo..."
python models/train_cnn3d.py

# 7. Evaluar modelo
echo "Paso 7: Evaluando modelo..."
python -m evaluation.evaluate

echo "=== Pipeline completado ==="

