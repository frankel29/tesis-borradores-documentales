## Configuración del Proyecto

Este módulo utiliza **FastText** para el procesamiento de lenguaje natural y la vectorización de documentos.

### 1. Requisitos de Datos (Modelo de Lenguaje)
Debido a su tamaño, el modelo pre-entrenado de vectores no se encuentra en este repositorio. Es necesario descargarlo manualmente:

* **Archivo:** `cc.es.300.bin`
* **Fuente:** [FastText - Spanish Vectors](https://fasttext.cc/docs/en/crawl-vectors.html)
* **Ubicación:** Debe colocarse en la raíz del proyecto (`/pruebas/`) para que `extractor.py` pueda cargarlo correctamente.

### 2. Instalación del Entorno
Para ejecutar los scripts, sigue estos pasos en tu terminal:

```bash
# Crear el entorno virtual
python -m venv .venv

# Activar el entorno
# En Windows:
.\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
