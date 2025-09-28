## Título y descripción breve

**Nombre del proyecto:** 
Detección de violencia armada en video vigilancia usando una arquitectura híbrida 3DCNN y LSTM

**Objetivo:** 
Desarrollar e implementar un sistema integral que, mediante un modelo híbrido **3DCNN-LSTM** y una interfaz web, detecte automáticamente **violencia armada** en videos para su **monitoreo en tiempo real**.


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

robo_armado_pipeline/
│
├── configs/
│   └── config.yaml               # Hiperparámetros y rutas
│
├── checkpoints/ 
│   ├── best_model.pth            # Guarda del mejor modelo 
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
├── eda/
│   ├── eda_test.py	              # Resumen general videos de prueba, distribución de clases, propiedades técnicas e histogramas para los datos de prueba
│   └── eda_train.py		      # Resumen general videos de prueba, distribución de clases, propiedades técnicas, histogramas y exploración frames de datos de entrenamiento
│
├── models/
│   ├── cnn3d_model.py            # Red neuronal 3D
│   └── train_cnn3d.py            # Entrenamiento usando los videos indexados
│
├── evaluation/
│   ├── evaluate.py               # Carga y evalúa el mejor modelo almacenado con el conjunto de prueba
│
├── logs/ 
│   └── training.log              # Log de entrenamiento: loss y accuracy por época
│
├── outputs/
│   ├── metrics/
│   │   ├── detailed_results.csv   # Resultados de predicción por clase con probabilidades asociadas
│   │   └── model_metrics.csv      # Métricas globales del modelo: accuracy, precision, recall y F1-score
│   └── plots/
│   │   ├── confusion_matrix.png                       # Matriz de confusión del modelo
│   │   ├── distribucion_clases_test.png               # Distribución de clases en el conjunto de prueba
│   │   ├── distribucion_clases_train.png              # Distribución de clases en el conjunto de entrenamiento
│   │   ├── distribucion_de_duracion_videos_test.png   # Duración de videos en el conjunto de prueba
│   │   ├── distribucion_de_duracion_videos_train.png  # Duración de videos en el conjunto de entrenamiento
│   │   ├── distribucion_de_fps_test.png               # Distribución de FPS en videos del conjunto de prueba
│   │   └── distribucion_de_fps_train.png              # Distribución de FPS en videos del conjunto de entrenamiento
│
├── requirements.txt	  # Listado de dependencias utilizadas en el proyecto.
└── run_pipeline.sh		  # Ejecución del pipeline completo


## Cómo correr el pipeline:

Para ejecutar el pipeline, primero instale las dependencias listadas en el archivo requirements.txt. Luego, ubicándose en la carpeta robo_armado_pipeline/, ejecute el script run_pipeline.sh para iniciar el proceso. 


## Resultados esperados: 

El pipeline ejecuta un modelo base y genera una organización completa de resultados para facilitar el análisis del rendimiento. En la ruta outputs/metrics se almacenan las principales métricas de evaluación accuracy, precision, recall y f1-score, mientras que en la carpeta outputs/plots se crean visualizaciones en formato PNG que incluyen la matriz de confusión, distribución de clases, distribución de duración de videos y distribución de FPS, todo ello disponible tanto para el conjunto de entrenamiento como para el de prueba.
Asimismo, el sistema registra todo el proceso de entrenamiento en el archivo logs/training.log
