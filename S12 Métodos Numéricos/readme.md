## Sprint 12 Métodos numéricos (Predicción de Precios de Coches)

### Descripción del Proyecto

El proyecto tiene como objetivo predecir los precios de coches usados basándose en características como el modelo, el año de fabricación, el kilometraje y otros factores relevantes. Utilizando técnicas de regresión, buscamos estimar el precio de venta de los coches y proporcionar recomendaciones basadas en estos datos.

### Objetivo

Desarrollar un modelo predictivo para estimar el precio de los coches usados, ayudando así a los compradores y vendedores a tomar decisiones informadas sobre la compra y venta de vehículos.

### Etapas del Proyecto

1. **Importación y Exploración de Datos**

   - Lectura de datos sobre coches usados de un archivo CSV.
   - Exploración inicial para identificar la distribución de precios y las características importantes.

2. **Preprocesamiento de Datos**

   - **Codificación de Características**: Aplicación de Binary Encoder para convertir características categóricas en valores binarios.
   - **Segmentación de Datos**: División de datos en conjuntos de entrenamiento (70%) y validación (30%).
   - **Escalamiento de Datos**: Normalización de las características numéricas para mejorar la performance del modelo.

3. **Entrenamiento del Modelo**

   - **Regresión Lineal**: Entrenamiento del modelo de regresión lineal utilizando el conjunto de entrenamiento.
   - **Evaluación del Modelo**: Validación del modelo con el conjunto de datos de prueba.

4. **Evaluación de Resultados**

   - **Precio Promedio Predicho**: 4598.69 euros.
   - **Error Cuadrático Medio (RMSE)**: 3080.16 euros, lo que indica la desviación promedio entre las predicciones y los valores reales.

### Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal.
- **Pandas**: Manipulación y análisis de datos.
- **NumPy**: Cálculos numéricos.
- **Scikit-Learn**: Implementación de modelos de Machine Learning.
- **Matplotlib**: Visualización de datos.

### Habilidades Relacionadas a Ciencia de Datos Desarrolladas

- **Análisis Exploratorio de Datos (EDA)**: Exploración y comprensión de la distribución de precios y características de los coches.
- **Preprocesamiento de Datos**: Limpieza, codificación, segmentación y escalamiento de datos.
- **Modelado Predictivo**: Entrenamiento y evaluación de modelos de regresión lineal.
- **Evaluación de Modelos**: Medición de la precisión del modelo mediante métricas como el RMSE.

### Archivos del Proyecto

- **Notebook**: Contiene el análisis completo y el código del proyecto.
- **Dataset**: Archivo CSV con datos de coches usados utilizados en el análisis.
- **Resultados**: Documentación con las conclusiones y recomendaciones basadas en el análisis realizado.

### Contenido de Archivos

- **.gitignore**: Plantilla de archivos a ignorar.
- **README.md**: Descripción del proyecto, herramientas utilizadas y detalles del análisis.
