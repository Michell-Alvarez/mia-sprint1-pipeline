## Título y descripción breve

**Nombre del proyecto:** 
Detección de violencia armada en video vigilancia usando una arquitectura híbrida 3DCNN.

**Objetivo:** 
Desarrollar e implementar un sistema integral que, mediante un modelo híbrido **3DCNN** y una interfaz web, detecte automáticamente **violencia armada** en videos para su **monitoreo en tiempo real**.


## Autores

**Michell Alvarez Cadillo** 
michell.alvarez.c@uni.pe 
[GitHub](https://github.com/Michell-Alvarez/)

**Victor Henostroza Villacorta** 
victor.henostroza.v@uni.pe
[GitHub](https://github.com/usuario2)


## Dataset

**Breve descripción:** 
El conjunto de datos utilizado contiene **266 videos** en total, organizados en dos clases principales: 
- **Violencia** (robos y agresiones armadas) 
- **No violencia** (escenas cotidianas sin incidentes violentos)

Se han dividido en los siguientes subconjuntos: 
- **Entrenamiento:** 212 videos (106 por clase) 
- **Prueba:** 54 videos (27 por clase)

**Nota:** Por motivos de **privacidad y confidencialidad**, el dataset **no puede ser compartido públicamente**, ya que contiene material sensible, incluyendo algunos videos vinculados a investigaciones policiales.

**Variables principales:** 
- `video_path`: ruta local del archivo 
- `label`: etiqueta numérica (0 = No Violencia, 1 = Violencia) 
- `class_name`: nombre de la clase 
- `filename`: nombre del video 
- `split`: partición (train / test)

**Fecha y hash de la versión usada:** 
- **Fecha:** septiembre de 2025 
- **Hash (versión interna):** `v1.0-2025-09-dataset-privado`


## Requisitos:

Ejecuta el siguiente comando para instalar las dependencias necesarias:
pip install -r requirements.txt


## Estructura del repositorio
```
robo_armado_pipeline/
│
├── configs/
│   ├──  config_baseline_fe_off.yaml    # Hiperparámetros y rutas modelo CNN base sin feature engineering
│   ├──  config_baseline_fe_on.yaml     # Hiperparámetros y rutas modelo CNN base con feature engineering
│   ├──  config_solid_fe_off.yaml       # Hiperparámetros y rutas modelo CNN+Atención temporal sin feature engineering
│   └──  config_solid_fe_on.yaml        # Hiperparámetros y rutas modelo CNN+Atención temporal con feature engineering
│
├── data/
│   ├── test_index.csv            # Índice de videos con ruta, etiqueta, clase y partición de datos de prueba
│   ├── train_index.csv           # Índice de videos con ruta, etiqueta, clase y partición de datos de entrenamiento
│   └── video_index.csv           # Índice de videos con ruta, clase y división train/test
│
├── data_ingest/
│   ├── create_index.py           # Escanea carpeta de videos y genera un CSV con rutas + etiqueta
│   ├── split_dataset.py          # Divide rutas para train/test
│   └── verify_structure.py	      # Verifica estructura de las carpetas	
│
├── detection/
│   ├── feature_engineering.py    # Extracción de características: LBP, color, combinación y verificación de pistola
│   └── yolov8_detection.py       # Detección y recorte de clips con YOLOv8
│
├── eda/
│   ├── metadata/
│   │   ├── test_videos.csv       # Metadatos: frames, fps, ancho, alto y duración de los videos del conjunto de prueba
│   │   ├── train_test_videos.csv # Metadatos: frames, fps, ancho, alto y duración de los videos del conjunto de entrenamiento+prueba
│   │   └── train_videos.csv      # Metadatos: frames, fps, ancho, alto y duración de los videos del conjunto de entrenamiento
│   └── plots/
│   │   ├── distribucion_clases_consolidado.png        # Distribución de clases en el conjunto de entrenamiento+prueba
│   │   ├── distribucion_clases_test.png               # Distribución de clases en el conjunto de prueba
│   │   ├── distribucion_clases_train.png              # Distribución de clases en el conjunto de entrenamiento
│   │   ├── distribucion_de_duracion_videos_consolidado.png         # Duración de videos en el conjunto de entrenamiento+prueba
│   │   ├── distribucion_de_duracion_videos_test.png   # Duración de videos en el conjunto de prueba
│   │   ├── distribucion_de_duracion_videos_train.png  # Duración de videos en el conjunto de entrenamiento
│   │   ├── distribucion_de_fps_consolidado.png        # Distribución de FPS en videos del conjunto de entrenamiento+prueba
│   │   ├── distribucion_de_fps_test.png               # Distribución de FPS en videos del conjunto de prueba
│   │   ├── distribucion_de_fps_train.png              # Distribución de FPS en videos del conjunto de entrenamiento
│   │   └── muestra_representativa_visualización_de_frames.jpeg     # frames de videos representativo
│   ├── eda_test.py	              # Resumen general videos de prueba, distribución de clases, propiedades técnicas e histogramas de los datos de prueba
│   ├── eda_train.py		      # Resumen general videos de entrenamiento, distribución de clases, propiedades técnicas, histogramas de datos de entrenamiento
│   └── eda_train_test.py		  # Resumen general videos de entrenamiento+prueba, distribución de clases, propiedades técnicas, histogramas de datos de entrenamiento+prueba
│
├── evaluation/
│   ├── evaluate_baseline.py      # Evalúa el modelo CNN base almacenado con el conjunto de prueba
│   └── evaluate_solid.py         # Evalúa el modelo CNN+Atención temporal almacenado con el conjunto de prueba
│
├── mlruns/                       # Registro local de experimentos y artefactos de MLflow
│
├── models/
│   ├── analyze_optuna.py         # Script de resultados de experimentos Optuna
│   ├── cnn3d_model_baseline.py   # Red neuronal 3D modelo CNN base
│   ├── cnn3d_model_solid.py      # Red neuronal 3D modelo CNN+Atención temporal
│   ├── train_cnn3d_baseline.py   # Entrenamiento modelo CNN base
│   ├── optimize.py               # Script de optimización de hiperparámetros con Optuna integrado a Mlflow
│   └── train_cnn3d_solid.py      # Entrenamiento modelo CNN+Atención temporal
│
# Estructura general de resultados (aplicable a cualquier variante de modelo)
# -------------------------------------------------------------------------
# Los parámetros principales son:
# - baseline_fe_off_0:  Modelo CNN base sin feature engineering
# - baseline_fe_on_1:   Modelo CNN base con feature engineering (color+textura)
# - baseline_fe_on_2:   Modelo CNN base con feature engineering (color)
# - baseline_fe_on_3:   Modelo CNN base con feature engineering (textura)
# - solid_fe_off_0:     Modelo CNN+Atención temporal sin feature engineering
# - solid_fe_on_1:      Modelo CNN+Atención temporal con feature engineering (color+textura)
# - solid_fe_on_2:      Modelo CNN+Atención temporal con feature engineering (color)
# - solid_fe_on_3:      Modelo CNN+Atención temporal con feature engineering (textura)
#
# Dentro de cada modelo se generan carpetas con estructura año-mes-día-hora-minuto-segundo,
# que contienen los resultados de evaluación, logs, métricas y el mejor modelo guardado.
├── outputs/
│   ├── baseline_fe_off/          # Ruta de ejemplo (idéntica estructura para los demás modelos)
│   │   ├── yyyy-mm-dd_hh-mm-ss   # Carpeta creada automáticamente por fecha y hora de ejecución
│   │   │   ├── eval/
│   │   │   │   ├── yyyy-mm-dd_hh-mm-ss/
│   │   │   │   │   ├── metrics/
│   │   │   │   │   │   ├── detailed_results.csv     # Predicciones por clase y probabilidades
│   │   │   │   │   │   └── model_metrics.csv        # Accuracy, precisión, recall, F1-score (test)
│   │   │   │   │   ├── plots/
│   │   │   │   │   │   └── confusion_matrix.png     # Matriz de confusión del conjunto de prueba
│   │   │   ├── logs/
│   │   │   │   └── training.log                     # Registro de entrenamiento: loss y accuracy por época
│   │   │   ├── metrics/
│   │   │   │   └── experiment_summary.log           # Modelo, run_id, seed y mejor accuracy obtenido
│   │   │   └── models/
│   │   │   │   └── best_model.pth                   # Mejor modelo guardado (según validación)
│
├── pruebas_externas/
|   |   ├── resultados/
│   │   │   ├── resultados_detec_videos_baseline.xlsx   # Comparación del rendimiento del modelo en videos locales (Offline)
│   │   │   ├── resultados_detec_webcam_baseline.xlsx   # Comparación del rendimiento del modelo en tiempo real (Webcam)
│   │   │   ├── robustez_zona_con_presencia_arma.xlsx   # Registro de frames procesados y el desempeño del modelo mediante verdaderos positivos (TP) y falsos negativos (FN).
│   │   │   └── robustez_zona_segura.xlsx               # Registro del rendimiento del modelo por video, incluyendo latencias, FPS y falsos positivos (FP) por minuto.
|   |   ├── videos/
│   │   │   ├── eval_rendimiento_videos.py      # Script que mide FPS y latencia total para videos locales 
│   │   │   ├── robustez_zona_presencia_arma.py	# Script que evalúa la robustez en zona con presencia de arma midiendo verdaderos positivos (TP) y falsos negativos (FN)             
│   │   │   └── robustez_zona_segura.py	        # Script que evalúa la robustez en zona segura midiendo falsos positivos (FP) y (FP por minuto)
|   |   └── webcam/
│   │   │   └── eval_rendimiento_realtime.py    # Script que mide FPS y latencia total en tiempo real  
│
├── tools/
|   |   ├── comparar_modelos.py     # Generar las métricas para los modelos CNN base y CNN+Atención temporal
|   |   └── mlflow_utils.py         # Funciones utilitarias para la carga, registro y gestión de experimentos con **MLflow**
│
├── readme.md	          # Archivo readme.md
├── requirements.txt	  # Listado de dependencias utilizadas en el proyecto.
└── run_pipeline.sh		  # Ejecución del pipeline completo


## Cómo correr el pipeline:

Para ejecutar el pipeline, primero instale las dependencias listadas en el archivo requirements.txt. Luego, ubicándose en la carpeta robo_armado_pipeline/, ejecute el script run_pipeline.sh para iniciar el proceso, los parámetros para su ejecución son:

./run_pipeline.sh [fe_off | fe_on] [baseline | solid] [0 | 1 | 2 | 3]

## Optuna búsqueda Bayes (TPE)
python models/optimize.py \
  --config configs/config_solid_fe_on.yaml \
  --optuna_name optuna_solid_fe_on_1 \
  --sampler tpe \
  --pruner median \
  --direction maximize \
  --n_trials 20


## Resultados esperados: 

El pipeline genera una estructura organizada de resultados que facilita el análisis del rendimiento. En la ruta outputs/eval/metrics se almacenan las principales métricas de evaluación, como accuracy, precision, recall y f1-score. Además, en outputs/eval/plots se generan las visualizaciones en formato PNG, entre ellas la matriz de confusión, mientras que la carpeta outputs/eda/plots contiene gráficos exploratorios como la distribución de clases, la duración de los videos y los valores de FPS. Finalmente, el sistema registra todo el proceso de entrenamiento en el archivo outputs/logs/training.log, permitiendo un seguimiento detallado y reproducible de cada ejecución.
