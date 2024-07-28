## Sprint 9 Aprendizaje supervisado

# Descripción del Proyecto

Este proyecto se centra en la implementación y evaluación de modelos de aprendizaje supervisado para predecir si un cliente de un banco cerrará su cuenta (churn). Se utilizan técnicas de preprocesamiento de datos, selección de características y varios algoritmos de clasificación para construir modelos predictivos.

## Objetivo

El objetivo principal es construir y evaluar modelos de aprendizaje supervisado que puedan predecir con precisión la probabilidad de que un cliente cierre su cuenta bancaria. Se espera que los modelos ayuden al banco a identificar clientes en riesgo y tomar medidas para retenerlos.

## Etapas del Proyecto

### Exploración y Entendimiento de los Datos

- Carga de datos y visualización de su estructura.
- Análisis descriptivo para comprender las características y distribución de los datos.

### Preprocesamiento de Datos

- Normalización de nombres de columnas por convención (snake case).
- Manejo de valores faltantes, específicamente en la columna 'Tenure'.
- Eliminación de columnas irrelevantes ('RowNumber', 'CustomerId', 'Surname').
- Codificación One-Hot para columnas categóricas ('Geography' y 'Gender').
- Escalado de características numéricas ('CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary').

### División de los Datos

- Separación de los datos en conjuntos de entrenamiento y prueba.
- Balanceo del conjunto de entrenamiento para manejar el desequilibrio de clases.

### Entrenamiento y Evaluación de Modelos

- Implementación de varios algoritmos de clasificación (Regresión Logística, K-Nearest Neighbors, Support Vector Machines, Random Forest, Gradient Boosting).
- Evaluación del rendimiento de los modelos utilizando métricas adecuadas (precisión, recall, F1-score, AUC-ROC).

### Optimización de Modelos

- Ajuste de hiperparámetros para mejorar el rendimiento de los modelos.
- Selección del modelo final basado en el rendimiento evaluado.

## Tecnologías Utilizadas

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Habilidades Relacionadas a Ciencia de Datos Desarrolladas

- **Exploración y Visualización de Datos:** Análisis descriptivo y visualización para comprender los datos y sus características.
- **Preprocesamiento de Datos:** Limpieza y transformación de datos, manejo de valores faltantes y codificación de variables categóricas.
- **Entrenamiento de Modelos:** Implementación y evaluación de varios algoritmos de aprendizaje supervisado.
- **Evaluación de Modelos:** Uso de métricas para evaluar y comparar el rendimiento de los modelos.
- **Optimización de Modelos:** Ajuste de hiperparámetros y selección del modelo final.

## Conclusiones y Resultados

- Se logró construir varios modelos de clasificación para predecir el churn de clientes.
- El modelo de Gradient Boosting mostró el mejor rendimiento con una alta precisión y recall.
- Las características más importantes para predecir el churn fueron 'Age', 'Balance' y 'NumOfProducts'.
- El banco puede utilizar estos modelos para identificar clientes en riesgo y diseñar estrategias de retención efectivas.

## Descripción de Archivos

- **Notebooks:** Contiene el Jupyter notebook con el análisis y desarrollo del proyecto.
- **Dataset:** Archivo con los datos utilizados para el entrenamiento y evaluación de los modelos.
