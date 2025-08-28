# Sentiment Analysis v2 🚀

Un sistema completo de análisis de sentimientos utilizando **LightGBM** y **Sentence Transformers** con una arquitectura moderna que incluye API REST, caching inteligente y optimizaciones de rendimiento.

## 📋 Tabla de Contenidos

- [Características](#características)
- [Arquitectura](#arquitectura)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [API Endpoints](#api-endpoints)
- [Modelos](#modelos)
- [Procesamiento de Datos](#procesamiento-de-datos)
- [Desarrollo](#desarrollo)
- [Testing](#testing)
- [Contribución](#contribución)
- [Licencia](#licencia)

## ✨ Características

### 🎯 Análisis de Sentimientos

- **3 clases de sentimiento**: Positivo (1), Negativo (0), Neutral (2)
- **Modelo LightGBM** optimizado con hyperparameter tuning
- **Embeddings** generados con `sentence-transformers/all-MiniLM-L6-v2`
- **Alta precisión** en clasificación multiclase

### 🔧 Arquitectura Moderna

- **API REST** construida con FastAPI
- **Caching inteligente** para embeddings y modelos
- **Procesamiento en paralelo** para datasets grandes
- **Optimizaciones de memoria** y rendimiento

### 🚀 Funcionalidades Avanzadas

- **Caching automático** de modelos en disco
- **Batch processing** optimizado para múltiples predicciones
- **Middleware CORS** para integración frontend
- **Logging detallado** para debugging y monitoreo
- **Health checks** y endpoints de testing

## 🏗️ Arquitectura

```
┌─────────────────┐      ┌────────────────┐       ┌─────────────────┐
│   Frontend         │       │      FastAPI      │       │      ML Pipeline   │
│   (Gradio)         │ ◄──► │    REST API       │ ◄──► │     ightGBM        │
│                    │       │                   │       │    + Embeddings.   │ 
└─────────────────┘      └────────────────┘       └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │       Data Layer   │
                    │       Caching &    │
                    │      Preprocessing │
                    └─────────────────┘
```

## 📦 Instalación

### Prerrequisitos

- Python 3.8+
- pip o conda
- Git

### Configuración del Entorno

1. **Clonar el repositorio**

```bash
git clone https://github.com/JUANJO-MDG/Sentiment-Analysis-v2-embeddings.git
cd "Sentiment Analysis v2"
```

2. **Crear entorno virtual**

```bash
python -m venv .venv
source .venv/bin/activate  # En Linux/Mac
# o
.venv\Scripts\activate     # En Windows
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

### Dependencias Principales

- **FastAPI** (0.116.1) - Framework web moderno
- **LightGBM** (4.6.0) - Modelo de gradient boosting
- **sentence-transformers** (5.1.0) - Generación de embeddings
- **scikit-learn** (1.7.1) - Métricas y evaluación
- **pandas** (2.3.2) - Manipulación de datos
- **uvicorn** (0.35.0) - Servidor ASGI

## 🚀 Uso

### API REST

1. **Iniciar el servidor**

```bash
python src/api/main.py
```

El servidor se ejecutará en `http://localhost:8000`. También puedes ejecutar el servidor de forma sencilla junto con la interfaz de gradio gradio en `http://localhost:8000/gradio`

2. **Hacer predicciones**

```bash
# Health check
curl http://localhost:8000/model/api/v2/health

# Test endpoint
curl http://localhost:8000/model/api/v2/test

# Predicción
curl -X POST "http://localhost:8000/model/api/v2/predict" \
     -H "Content-Type: application/json" \
     -d '{"message": "I love this product!"}'
```

### Uso Programático

```python
from src.model.prediction_service import get_prediction_service

# Obtener servicio (singleton con caching)
service = get_prediction_service()

# Predicción individual
label, score = service.predict_sentiment("This is amazing!")
print(f"Sentiment: {label} (score: {score})")

# Predicción por lotes (optimizada)
texts = ["I love it!", "This is terrible", "It's okay"]
results = service.predict_batch_optimized(texts)
```

### Testing del Modelo

(TESTING)

## 📁 Estructura del Proyecto

```
Sentiment Analysis v2/
├── 📁 src/                          # Código fuente principal
│   ├── 📁 api/                      # API REST con FastAPI
│   │   ├── main.py                  # Aplicación principal
│   │   ├── 📁 routes/               # Endpoints de la API
│   │   │   └── route.py             # Rutas de predicción
│   │   └── 📁 schemas/              # Modelos de datos Pydantic
│   │       └── model_schemas.py     # Esquemas request/response
│   │
│   ├── 📁 model/                    # Lógica de modelos ML
│   │   ├── model.py                 # Creación y configuración LightGBM
│   │   └── prediction_service.py    # Servicio de predicción con caching
│   │
│   ├── 📁 models/                   # Gestión de modelos y embeddings
│   │   ├── utils.py                 # Métricas de evaluación
│   │   └── 📁 embeddings/           # Gestión de embeddings
│   │       ├── emb_model.py         # Modelo de embeddings con caching
│   │       └── emb_data.py          # Carga y generación de embeddings
│   │       📁 predictor/
│   │           └── lgbm_sentiment_predictor.joblib  # Modelo LightGBM
│   │
│   ├── 📁 data/                     # Procesamiento de datos
│   │   └── preprocessing.py         # Limpieza y procesamiento de texto
│   │
│   │
│   └── 📁 frontend/                 # Interfaz web
│       └── gr_interface.py          # Interfaz Gradio (TODO)
│
├── 📁 models_cache/                 # Cache de modelos embeddings
├── 📁 data/                         # Datasets y datos procesados
├── 📁 tests/
├── .dockerignore                    # Archivos ignorados por Docker
├── Dockerfile                       # Instrucciones Docker
├── requirements.txt                 # Dependencias Python
├── .gitignore                       # Archivos ignorados por Git
└── README.md                        # Este archivo
```

## 🔌 API Endpoints

### Base URL: `http://localhost:8000/model/api/v2`

| Endpoint   | Método | Descripción                         | Request                | Response                                                 |
| ---------- | ------ | ----------------------------------- | ---------------------- | -------------------------------------------------------- |
| `/health`  | GET    | Health check del servicio           | -                      | `{"status": "healthy", "service": "sentiment-analysis"}` |
| `/test`    | GET    | Test automático con mensaje ejemplo | -                      | `{"status": "success", "prediction": "...", "score": 1}` |
| `/predict` | POST   | Predicción de sentimiento           | `{"message": "texto"}` | `{"sentiment": "Positive\|Negative\|Neutral"}`           |

### Ejemplos de Uso

```javascript
// JavaScript/Frontend
fetch("http://localhost:8000/model/api/v2/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    message: "I absolutely love this new feature!",
  }),
})
  .then((response) => response.json())
  .then((data) => console.log(data.sentiment));
```

```python
# Python
import requests

response = requests.post(
    'http://localhost:8000/model/api/v2/predict',
    json={'message': 'This product is terrible'}
)
print(response.json()['sentiment'])  # "Negative"
```

## 🤖 Modelos

### LightGBM Classifier

- **Objetivo**: Clasificación multiclase (3 clases)
- **Métrica**: Multi-class log-loss
- **Características**: Optimizado con RandomizedSearchCV
- **Archivo**: `src/models/predictor/lgbm_sentiment_predictor.joblib`

### Sentence Transformer

- **Modelo**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensiones**: 384 features por embedding
- **Caching**: Automático en disco para evitar re-descargas
- **Ubicación Cache**: `models_cache/all-MiniLM-L6-v2/`

### Rendimiento

- **Precisión**: 75% (dependiendo del dataset)
- **Velocidad**:
  - Primera predicción: ~2-3s (carga de modelos)
  - Predicciones siguientes: ~50-100ms
  - Batch processing: ~10-20ms por texto

## 📊 Procesamiento de Datos

### Pipeline de Limpieza

1. **Conversión a minúsculas**
2. **Eliminación de URLs** (`http://`, `www.`, `https://`)
3. **Eliminación de menciones** (`@usuario`, `#hashtag`)
4. **Eliminación de números**
5. **Eliminación de emojis**
6. **Eliminación de puntuación**
7. **Normalización de espacios en blanco**

### Optimizaciones

- **Patrones RegEx pre-compilados** para máximo rendimiento
- **Procesamiento en paralelo** utilizando multiprocessing
- **Batch processing** para embeddings
- **Caching de embeddings** en formato NumPy

### Formato de Datos

```python
# Dataset esperado
df = pd.DataFrame({
    'Text': ['I love this!', 'This is bad', 'It\'s okay'],
    'Label': [1, 0, 2]  # 1=Positive, 0=Negative, 2=Neutral
})
```

## 🛠️ Desarrollo

### Configuración del Entorno de Desarrollo

```bash
# Instalar dependencias de desarrollo
pip install -r requirements.txt

# Configurar pre-commit hooks (opcional)
pip install pre-commit
pre-commit install
```

### Estructura de Desarrollo

- **Logging**: Configurado en todos los módulos principales
- **Error Handling**: HTTPException para API, try-catch en servicios
- **Type Hints**: Utilizados en funciones críticas
- **Docstrings**: Documentación completa en estilo Google

### Variables de Entorno

```bash
# .env (opcional)
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
CACHE_DIR=models_cache
```

### Agregar Nuevas Funcionalidades

1. **Nuevo Endpoint**:

   - Agregar ruta en `src/api/routes/route.py`
   - Crear schema en `src/api/schemas/model_schemas.py`
   - Documentar en este README

2. **Nuevo Modelo**:
   - Agregar lógica en `src/model/`
   - Integrar con `prediction_service.py`
   - Actualizar tests

## 🧪 Testing

### Pruebas unitarias

(`test_api.py`)

1. En este test se hacen tests con `test_e2e_prediction()` Y con `test_health_endpoint()` para ver que la respuesta del modelo sea la esperada.

(`test_interface.py`) 2. En este test se busca que al hacer una prediccion nos de alguna de estas etiquetas Positive, Negative o Neutral.

(`test_model`) 3. En este test se busca que las predicciones del modelo sean correctas y validas.

## 🔧 Troubleshooting

### Problemas Comunes

1. **Error de numpy.int64 serialization**

   - **Causa**: FastAPI no puede serializar tipos NumPy
   - **Solución**: Ya implementado - conversión a `int()` nativo

2. **Modelos no se cargan**

   - **Causa**: Rutas incorrectas o archivos faltantes
   - **Solución**: Verificar estructura de carpetas y paths

3. **Memoria insuficiente**

   - **Causa**: Datasets muy grandes
   - **Solución**: Reducir `batch_size` en procesamiento

4. **Lentitud en primera predicción**
   - **Causa**: Descarga/carga inicial de modelos
   - **Solución**: Normal, predicciones siguientes serán rápidas

### Logs de Debug

```bash
# Ver logs detallados
export LOG_LEVEL=DEBUG
python src/api/main.py
```

## 📈 Métricas y Monitoreo

### Métricas del Modelo

- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase (weighted)
- **Recall**: Recall por clase (weighted)
- **F1-Score**: Score F1 (weighted)
- **Confusion Matrix**: Matriz de confusión

### Uso de Métricas

```python
from src.models.utils import model_metrics

# y_true: etiquetas reales
# y_pred: predicciones del modelo
model_metrics(y_true, y_pred)
```

## 🚧 Roadmap

### Próximas Funcionalidades

- [ ] **Interfaz Gradio** completamente funcional
- [ ] **Balanceado de datasets** automático
- [ ] **Soporte para más idiomas**
- [ ] **API de entrenamiento** para modelos personalizados
- [ ] **Métricas en tiempo real** con dashboard
- [ ] **Docker containerization**
- [ ] **Deployment scripts** para producción

### Mejoras Técnicas

- [ ] **Tests unitarios** completos
- [ ] **CI/CD pipeline** con GitHub Actions
- [ ] **Monitoreo de performance** con Prometheus
- [ ] **Rate limiting** en API
- [ ] **Autenticación** y autorización
- [ ] **Versionado de modelos** con MLflow

## 🤝 Contribución

### Cómo Contribuir

1. Fork el repositorio
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit los cambios (`git commit -am 'Add nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### Guías de Desarrollo

- Seguir PEP 8 para estilo de código Python
- Agregar docstrings a todas las funciones públicas
- Incluir tests para nuevas funcionalidades
- Actualizar este README si es necesario

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Soporte

Si tienes problemas o preguntas:

1. Revisa la sección [Troubleshooting](#troubleshooting)
2. Busca en Issues existentes
3. Crea un nuevo Issue con detalles específicos
4. Incluye logs y pasos para reproducir el problema

---

**Desarrollado con ❤️ usando Python, FastAPI, LightGBM y Sentence Transformers**
