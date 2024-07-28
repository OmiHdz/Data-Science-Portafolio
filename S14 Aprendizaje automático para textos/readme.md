## Sprint 14 Aprendizaje automático para textos : Modelos de Clasificación de reseñas de películas en IMDB

### Descripción del Proyecto
En este sprint, se implementaron y evaluaron múltiples modelos avanzados de clasificación de texto, incluyendo LightGBM, Random Forest, Logistic Regression, y un modelo ensamblado con Voting Classifier. El objetivo principal fue optimizar el rendimiento del modelo en la clasificación de reseñas, utilizando técnicas de procesamiento de texto con spaCy y TF-IDF, así como BERT para la generación de embeddings.

### Objetivo
Desarrollar y evaluar modelos de clasificación de texto avanzados para mejorar el rendimiento en la clasificación de reseñas, con un enfoque en la precisión y capacidad de generalización del modelo. Se buscó alcanzar un F1 Score superior a 0.85 en el conjunto de prueba.

### Etapas del Proyecto
- **Preprocesamiento de Datos**: Normalización y limpieza de reseñas utilizando spaCy y TF-IDF.
- **Entrenamiento de Modelos**: Entrenamiento de modelos individuales como LightGBM, Random Forest, y Logistic Regression.
- **Ensamblaje de Modelos**: Implementación de un Voting Classifier para combinar las predicciones de los modelos individuales.
- **Uso de Embeddings de BERT**: Generación de embeddings para el texto utilizando BERT, y entrenamiento de un modelo de regresión logística sobre estos embeddings.
- **Evaluación**: Evaluación del rendimiento de los modelos utilizando métricas como Exactitud, F1 Score, APS y ROC AUC.

### Herramientas Tecnológicas Implementadas
- **LightGBM**: Para capturar relaciones complejas y no lineales.
- **Random Forest Classifier**: Robusto a sobreajuste y eficaz con datos de muchas características.
- **Logistic Regression**: Modelo simple y efectivo para relaciones lineales.
- **Voting Classifier**: Combina los modelos anteriores para mejorar el rendimiento general.
- **spaCy**: Procesamiento de texto para normalización.
- **TF-IDF**: Vectorización de texto.
- **BERT**: Para generar embeddings de texto y mejorar la capacidad de representación.

### Habilidades Relacionadas a Ciencia de Datos Desarrolladas
- **Implementación de Modelos Avanzados**: Aplicación de técnicas avanzadas como Voting Classifier y BERT en problemas de clasificación de texto.
- **Evaluación de Modelos**: Uso de métricas avanzadas para evaluar el rendimiento del modelo.
- **Optimización de Modelos**: Ajuste y combinación de modelos para mejorar el rendimiento.
