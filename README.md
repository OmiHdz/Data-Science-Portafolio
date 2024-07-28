# Data-Science-Portafolio
Portafolio de proyectos realizados en bootcamp Data Scientist de Tripleten

## Herramientas Tecnológicas Implementadas

En esta sección se resumen todas las herramientas y tecnologías utilizadas en los proyectos de este portafolio:

- **Lenguajes de Programación:**
  - **Python** 
  - **SQL:** Lenguaje de consulta para bases de datos.

- **Bibliotecas y Paquetes de Python:**
  - **Pandas:** Manipulación y análisis de datos.
  - **NumPy:** Cálculos numéricos.
  - **Matplotlib:** Visualización de datos.
  - **Seaborn:** Visualización estadística.
  - **Scikit-Learn:** Modelos y algoritmos de Machine Learning.
  - **TensorFlow:** Framework para el desarrollo y entrenamiento de redes neuronales.
  - **PyTorch:** Framework de Machine Learning y redes neuronales.
  - **SciPy:** Biblioteca para cálculos científicos y técnicos.
  - **Statsmodels:** Biblioteca para la estimación de modelos estadísticos.
  - **LightGBM:** Modelado eficiente para datos grandes y complejos.
  - **BERT:** Para generar embeddings de texto y mejorar la capacidad de representación.
  - **spaCy:** Procesamiento de texto para normalización.
  - **NLTK:** Toolkit para procesamiento del lenguaje natural.
  - **Plotly:** Visualización interactiva de datos.
  - **Geopandas:** Manipulación y análisis de datos geoespaciales.

- **Herramientas de Desarrollo y Entorno:**
  - **Jupyter Lab:** Entorno interactivo para el desarrollo y la visualización de datos.
  - **Anaconda:** Distribución de Python para ciencia de datos.
  - **Git:** Control de versiones.
  - **Visual Studio Code:** Editor de código.

- **Herramientas de Web Scraping y ETL:**
  - **Requests:** Biblioteca para realizar peticiones HTTP.
  - **BeautifulSoup:** Biblioteca para extraer datos de archivos HTML y XML.

- **Bases de Datos:**
  - **SQLite:** Sistema de gestión de bases de datos ligero.
  - **PostgreSQL:** Sistema de gestión de bases de datos relacional.




## Sprint 2 Python Básico
# [Proyecto de Análisis de Datos: Preferencias Musicales en Springfield y Shelbyville](https://github.com/OmiHdz/Data-Science-Portafolio/blob/5ca4a258d78582a3bf9d0e1c7094194b2fb64659/S2%20Python%20basico%202/P2_Python_basico_2.ipynb)

## Descripción del Proyecto
Como analista de datos, tu trabajo consiste en analizar datos para extraer información valiosa y tomar decisiones basadas en ellos. Este proyecto se enfoca en comparar las preferencias musicales de las ciudades de Springfield y Shelbyville, utilizando datos reales de transmisión de música online. El objetivo es probar la hipótesis de que la actividad de los usuarios y las usuarias varía según el día de la semana y dependiendo de la ciudad.

## Objetivo
Probar la hipótesis:
- La actividad de los usuarios y las usuarias difiere según el día de la semana y dependiendo de la ciudad.

## Etapas del Proyecto
1. **Descripción de los Datos:**
   - Evaluar la calidad y estructura de los datos.
   - Identificar problemas y obtener una comprensión general.

2. **Preprocesamiento de Datos:**
   - Limpiar y transformar los datos para preparar el análisis.
   - Ajustar los encabezados, manejar valores ausentes, eliminar duplicados y corregir errores en los datos.

3. **Prueba de Hipótesis:**
   - Analizar los datos para probar la hipótesis planteada.
   - Comparar la actividad musical en diferentes días de la semana y entre ciudades.

## Tecnologías Utilizadas
- **Python:** Lenguaje de programación principal.
- **Pandas:** Biblioteca utilizada para la manipulación y análisis de datos.
- **Jupyter Notebook:** Herramienta para documentar y ejecutar el análisis de datos de manera interactiva.

## Habilidades Relacionadas a Ciencia de Datos Desarrolladas
- **Limpieza y Preprocesamiento de Datos:** Identificación y corrección de problemas en los datos, manejo de valores ausentes y eliminación de duplicados.
- **Análisis Exploratorio de Datos:** Evaluación de la calidad de los datos y extracción de conclusiones iniciales.
- **Prueba de Hipótesis:** Formulación y prueba de hipótesis basadas en el análisis de datos.
- **Uso de Herramientas de Análisis de Datos:** Aplicación de bibliotecas y herramientas para manipular y analizar datos en Python.



## Sprint 3 Data Wrangling
# [Proyecto de Análisis de Datos: Comportamiento de Compras en una Plataforma de E-commerce](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S3%20Data%20Wrangling/P3_Data_Warangling.ipynb)

## Descripción del Proyecto
Este proyecto se enfoca en el análisis de datos de una plataforma de e-commerce para entender los patrones de compra de los clientes. A través del análisis de datos históricos, buscamos identificar tendencias en la hora del día y el día de la semana en que se realizan los pedidos, así como los productos más frecuentes y los patrones de reordenamiento.

## Objetivo
El objetivo principal es descubrir patrones y comportamientos en las compras de los clientes para optimizar las estrategias de marketing y mejorar la experiencia del usuario. También se pretende identificar los productos más populares y entender la frecuencia de compra de los clientes.

## Etapas del Proyecto

1. **Preprocesamiento de los Datos:**
   - Identificación y manejo de valores ausentes en los dataframes.
   - Reemplazo de valores ausentes por 'Unknown' y eliminación de duplicados innecesarios.
   - Normalización y conversión de tipos de datos para facilitar el análisis.

2. **Análisis de los Datos:**

   **[A] Verificación y Visualización de Patrones de Compra**
   - Comprobación de valores en columnas críticas como `order_hour_of_day` y `order_dow`.
   - Creación de gráficos para visualizar el número de pedidos por hora del día y por día de la semana.
   - Análisis del tiempo entre pedidos y su distribución.

   **[B] Comparación de Patrones entre Días Específicos**
   - Comparación de las horas de pedido entre miércoles y sábados.
   - Análisis de la distribución del número de pedidos por cliente.
   - Identificación de los 20 productos más frecuentes en pedidos.

   **[C] Análisis Detallado de Productos y Clientes**
   - Estudio del número de artículos comprados por pedido.
   - Identificación de los artículos que se reordenan con mayor frecuencia.
   - Cálculo de la tasa de repetición de pedido para productos y clientes.
   - Análisis de los artículos que se añaden primero al carrito.

## Tecnologías Utilizadas
- **Python 3.12**
- **Pandas:** Para la manipulación y análisis de datos.
- **Matplotlib:** Para la visualización de datos.
- **Jupyter Notebook:** Para la documentación y ejecución del análisis.

## Habilidades Desarrolladas
- **Limpieza y Preprocesamiento de Datos:** Manejo de valores ausentes y duplicados, normalización de datos.
- **Análisis Exploratorio de Datos (EDA):** Identificación de patrones y tendencias en datos de compras.
- **Visualización de Datos:** Creación de gráficos para interpretar datos de manera efectiva.
- **Análisis de Comportamiento del Cliente:** Comprensión de la frecuencia y patrones de compra de clientes.

## Conclusiones
- Los clientes tienden a realizar más compras los domingos y lunes, con una mayor actividad durante las mañanas y primeras horas de la tarde.
- La frecuencia de compra varía principalmente entre 7, 14 y 30 días.
- Los productos más vendidos son principalmente víveres, con las bananas siendo el más popular.
- La tasa de reorden para la mayoría de los productos oscila entre el 45% y el 55%.
- Los artículos que se reordenan con mayor frecuencia son consistentes con los productos más vendidos.




## Sprint 4 Análisis estadístico de datos (Modelo de recomendación de planes de tarifas telefónicas)

# [Proyecto de Análisis de Datos: Análisis Estadístico de Planes Telefónicos](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S4%20An%C3%A1lisis%20estad%C3%ADstico%20de%20datos/P4_Analisis_estadistico.ipynb)

## Descripción del Proyecto
El proyecto se centra en el análisis estadístico de dos planes telefónicos ofrecidos por una empresa. El objetivo es determinar la diferencia en los ingresos generados por los planes "Surf" y "Ultimate", así como comparar los ingresos de usuarios en diferentes áreas geográficas (NY-NJ vs. otras áreas). Se utilizan técnicas de análisis de datos y estadística para probar hipótesis sobre la rentabilidad y el comportamiento de los usuarios.

## Objetivo
- Determinar si los ingresos generados por los planes "Surf" y "Ultimate" son significativamente diferentes.
- Comparar los ingresos de usuarios en el área de NY-NJ con los de otras áreas.
- Evaluar la rentabilidad de los planes telefónicos desde diferentes perspectivas.

## Etapas del Proyecto
1. **Carga y Exploración de Datos:**
   - Importación de datos y bibliotecas necesarias.
   - Análisis exploratorio de datos (EDA) para entender la distribución y características de los datos.

2. **Análisis Estadístico:**
   - Pruebas de hipótesis para comparar ingresos entre los planes "Surf" y "Ultimate".
   - Pruebas de hipótesis para comparar ingresos entre usuarios del área NY-NJ y otras áreas.
   - Análisis de la rentabilidad de los planes desde diferentes perspectivas (fidelidad del cliente vs. redituabilidad).

3. **Conclusiones y Recomendaciones:**
   - Resumen de los resultados de las pruebas de hipótesis.
   - Recomendaciones basadas en los hallazgos del análisis.

## Tecnologías Utilizadas
- **Python 3.12**
- **Bibliotecas:**
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scipy
  - statsmodels

## Habilidades Relacionadas a Ciencia de Datos Desarrolladas
- **Análisis exploratorio de datos (EDA)**
- **Pruebas de hipótesis**
- **Visualización de datos**
- **Interpretación de resultados estadísticos**
- **Generación de recomendaciones basadas en datos**

## Análisis del Proyecto

### Carga y Exploración de Datos
Se importan los datos necesarios y se realiza un análisis exploratorio para comprender la distribución de los datos y sus características. Se visualizan las distribuciones de los ingresos por plan y por área geográfica utilizando gráficos de cajas y diagramas de densidad.

### Análisis Estadístico

**Pruebas de Hipótesis: Ingresos por Plan**
- Se utiliza la prueba de Mann-Whitney U para comparar los ingresos entre los planes "Surf" y "Ultimate".
- Se rechaza la hipótesis nula, lo que indica que los ingresos de los dos planes son significativamente diferentes.

**Pruebas de Hipótesis: Ingresos por Área Geográfica**
- Se utiliza la prueba de Mann-Whitney U para comparar los ingresos entre usuarios del área NY-NJ y otras áreas.
- Se rechaza la hipótesis nula, indicando que los ingresos de los usuarios de NY-NJ son significativamente diferentes a los de otras áreas.

### Conclusiones y Recomendaciones

**Conclusiones Generales:**
- Los ingresos generados por los planes "Surf" y "Ultimate" son significativamente diferentes.
- Los ingresos de los usuarios del área NY-NJ son significativamente diferentes a los de otras áreas.

**Recomendaciones:**
- Desde una perspectiva de fidelidad y satisfacción del cliente, podría ser útil un estudio de mercado para rediseñar el plan "Ultimate", reduciendo un 15% los precios y beneficios, manteniendo así la satisfacción del cliente y reduciendo el riesgo de penalizaciones.
- Desde una perspectiva de rentabilidad, el plan "Surf" es redituable debido a las penalizaciones generadas por los clientes. El plan "Ultimate" también es redituable debido al bajo aprovechamiento de los beneficios por parte de los usuarios.

## Visualizaciones
Se generaron varias visualizaciones para apoyar el análisis:
- Gráficos de cajas y diagramas de densidad para comparar distribuciones de ingresos.
- Histogramas para visualizar la frecuencia de ingresos en diferentes rangos.




## [Sprint 5 Herramientas de desarrollo de software: Análisis de Dataset de Ventas de Automóviles en Estados Unidos](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S5%20Herramientas%20de%20desarrollo%20de%20Software/notebooks/EDA.ipynb)

En este proyecto abordamos el análisis de un dataset de ventas de automóviles en Estados Unidos.

Utilizamos un histograma y un gráfico de dispersión que nos arrojan conclusiones, en ellas podemos observar:

- La mayor parte de los vehículos han recorrido entre 90,000 y 150,000 millas en su odómetro.
- Aunque existen excepciones, entre menos millas recorridas, mayor es el precio.
- Las excepciones posiblemente sean vehículos con buen millaje pero sean clásicos o de marcas con poca devaluación.

## ¿Cómo utilizarlo?

Para ejecutar el programa, es necesario marcar los checkboxes mostrados en el tablero. Estos, al ser seleccionados, mostrarán:

- Histograma que muestra la distribución de frecuencias del millaje recorrido.
- Diagrama de dispersión que muestra la relación entre el millaje recorrido y el precio.

**NOTA:** Este proyecto es acerca de la construcción y despliegue de una aplicación de ciencia de datos en un servidor web. Para este proyecto utilizamos herramientas y conocimientos como:

- Plotly Express
- Pandas
- Anaconda
- Visual Studio Code
- Entornos Virtuales
- Servicios Web
- Python
- Render

## Descripción de Archivos

- **notebooks (Carpeta):** Contiene el Jupyter notebook utilizado en el desarrollo del proyecto.
- **streamlit (Carpeta):** Contiene la configuración del servidor para Render.
- **.gitignore:** Contiene la plantilla de archivos a ignorar.
- **README.md:** Contiene la descripción del proyecto, herramientas utilizadas y descripción de archivos.
- **app.py:** Contiene el código a ejecutar, aplicación web.
- **requirements.txt:** Contiene los requisitos mínimos de librerías a instalar para poder ejecutar correctamente el código.
- **vehicles_us.csv:** Contiene el dataset referenciado en el proyecto.

## URL del Proyecto

[https://project-s5.onrender.com](https://project-s5.onrender.com)



## [Sprint 6 Predicción de ventas en tienda online (Proyecto integrado S1 - S5)](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S6%20Proyecto%20integrado%201/Jupyter%20Notebook/project_s6.ipynb)

En este proyecto abordamos el análisis de un dataset de ventas de videojuegos en 3 regiones: Norte América (NA) Unión Europea (EU) Japón (JP)

Este dataset contiene datos desde 1980 hasta el 2016, se pretende utilizar para hacer estimaciones hacia el 2017

Este dataset (games.csv) incluye datos de ventas, géneros, clasificación ESRB, región, nombre del videojuego y plataforma del videojuego.

Con estos datos realizamos el proceso ETL así como un análisis exploratorio que nos permitió llegar a conclusiones como:

* Preferencias de ventas por región
* Popularidad de las plataformas
* Ventas globales por género y plataforma
* Comportamiento de ventas de las diferentes plataformas
* Comportamiento del ciclo de vida de las plataformas
* Pruebas de hipótesis
* Correlación entre calificaciones de usuario y profesionales contra ventas globales
* Preferencias de género por región

Todos estos insights nos sirvieron para generar conclusiones y recomendaciones para una estrategia de venta y marketing para videojuegos para el año próximo (2017).

utilizamos herramientas como:

Python
Jupyter Notebooks
Anaconda
Pandas
Matplotlib
Scipy
Seaborn
Numpy
Visual Studio Code entre otras.
Descripción de archivos: Notebooks (Carpeta) : Contiene el Jupypter notebook así como el dataset utilizado en el desarrollo del proyecto

.gitignore: Contiene la plantilla de archivos a ignorar

Readme.md : Contiene la descripción del proyecto, herramientas utilizadas y descripción de archivos

requirements.txt : Contiene los requisitos mínimos de librerías a instalar para poder ejecutar correctamente el código



## [Sprint 7 Recopilación y almacenamiento de datos SQL (Series temporales para predecir el número de pedidos de taxis)](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S7_SQL/P7_Recopilaci%C3%B3n%20y%20almacenamiento%20de%20datos.ipynb)

## Descripción del Proyecto
Este proyecto se centra en la recopilación y almacenamiento de datos. Se abordan diferentes técnicas y herramientas para extraer datos de diversas fuentes y almacenarlos de manera eficiente para su posterior análisis.

## Objetivo
El objetivo principal de este proyecto es implementar un flujo de trabajo para la recopilación y almacenamiento de datos, utilizando herramientas y técnicas adecuadas para asegurar la integridad y accesibilidad de los datos.

## Etapas del Proyecto
1. **Identificación de Fuentes de Datos:** Selección de las fuentes de datos relevantes para el proyecto.
2. **Extracción de Datos:** Implementación de scripts y herramientas para la extracción de datos.
3. **Transformación de Datos:** Procesamiento y limpieza de los datos para asegurar su calidad.
4. **Almacenamiento de Datos:** Selección y uso de tecnologías de almacenamiento adecuadas.
5. **Documentación y Reporte:** Creación de documentación y reportes para describir el proceso y los resultados.

## Tecnologías Utilizadas
- **pandas:** Para la manipulación y análisis de datos.
- **requests:** Para la extracción de datos desde APIs.
- **sqlite3:** Para el almacenamiento de datos en una base de datos SQL.
- **json:** Para la manipulación de datos en formato JSON.
- **datetime:** Para el manejo de fechas y tiempos.

## Habilidades Relacionadas a Ciencia de Datos Desarrolladas
- **Web Scraping y APIs:** Técnicas de extracción de datos desde la web y APIs.
- **Procesamiento de Datos:** Limpieza y transformación de datos.
- **Almacenamiento de Datos:** Uso de bases de datos SQL para almacenar datos.
- **Documentación:** Creación de documentación clara y detallada sobre el proceso y resultados del proyecto.
- **Manejo de Fechas y Tiempos:** Manipulación y análisis de datos temporales.



## [Sprint 8 Introducción al Machine Learning](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S8%20Introducci%C3%B3n%20al%20Machine%20Learning/P9_Introducci%C3%B3n%20al%20Machine%20Learning.ipynb)

### Descripción del Proyecto

En este proyecto, se exploran los conceptos fundamentales del Machine Learning (ML) y su aplicación práctica. Se cubren técnicas y algoritmos esenciales utilizados en ML, junto con ejemplos prácticos y análisis de datos para ilustrar su uso.

### Objetivo

El objetivo del proyecto es proporcionar una comprensión sólida de los principios básicos del Machine Learning, sus aplicaciones y cómo implementar diferentes algoritmos para resolver problemas de análisis de datos.

### Etapas del Proyecto

1. **Introducción a Machine Learning**:
    - Definición y conceptos básicos.
    - Diferencias entre aprendizaje supervisado y no supervisado.
    - Importancia del preprocesamiento de datos.
2. **Análisis Exploratorio de Datos**:
    - Descripción de los datos y su origen.
    - Técnicas de visualización para entender mejor los datos.
    - Identificación de patrones y relaciones significativas en los datos.
3. **Preprocesamiento de Datos**:
    - Limpieza de datos y tratamiento de valores faltantes.
    - Normalización y escalamiento de datos.
    - Dividir los datos en conjuntos de entrenamiento y prueba.
4. **Implementación de Algoritmos de Machine Learning**:
    - Introducción a varios algoritmos de aprendizaje supervisado (regresión, clasificación).
    - Implementación de modelos de regresión lineal, regresión logística y árboles de decisión.
    - Evaluación del rendimiento de los modelos mediante métricas adecuadas.
5. **Aprendizaje No Supervisado**:
    - Explicación de algoritmos de agrupamiento (clustering).
    - Implementación de K-means y análisis de sus resultados.
6. **Conclusiones y Resultados**:
    - Resumen de hallazgos clave.
    - Discusión de la efectividad de los modelos implementados.
    - Recomendaciones para futuras mejoras y aplicaciones.

### Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal utilizado en el proyecto.
- **pandas**: Biblioteca para la manipulación y análisis de datos.
- **numpy**: Biblioteca para cálculos numéricos.
- **matplotlib.pyplot**: Biblioteca para la visualización de datos.
- **seaborn**: Biblioteca para la visualización de datos.
- **scikit-learn**: Biblioteca para la implementación de algoritmos de Machine Learning.

### Habilidades Relacionadas a Ciencia de Datos Desarrolladas

- **Análisis Exploratorio de Datos (EDA)**: Capacidad para explorar y visualizar datos para obtener información valiosa.
- **Preprocesamiento de Datos**: Técnicas para limpiar, normalizar y preparar datos para el análisis.
- **Implementación de Modelos de Machine Learning**: Habilidad para aplicar algoritmos de ML supervisado y no supervisado.
- **Evaluación de Modelos**: Capacidad para evaluar y comparar el rendimiento de diferentes modelos.
- **Visualización de Datos**: Uso de herramientas de visualización para interpretar y comunicar resultados.

### Modelos Utilizados

- **Regresión Lineal**
- **Regresión Logística**
- **Árboles de Decisión**
- **K-means (Clustering)**

## [Sprint 9 Aprendizaje supervisado](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S9%20Aprendizaje%20Supervisado/P9_Aprendizaje%20Supervisado.ipynb)

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
- **requirements.txt:** Lista de librerías necesarias para reproducir el análisis.


## [Sprint 10 Aprendizaje Automático en Negocios](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S10%20Aprendizaje%20autom%C3%A1tico%20en%20negocios/P10_Aprendizaje%20Automatico%20negocios.ipynb)

# Proyecto de Optimización de Perforación de Pozos Petrolíferos

## Descripción del Proyecto

El proyecto se centra en optimizar la perforación de 200 pozos de petróleo en la compañía OilyGiant. Utilizando datos de exploración geológica, nuestro objetivo es identificar las mejores regiones para la extracción de crudo y maximizar los beneficios esperados. El análisis incluye la predicción del volumen de reservas en nuevas perforaciones y la evaluación de riesgos y beneficios utilizando técnicas de bootstrapping.

## Objetivo

El objetivo principal es encontrar las mejores ubicaciones para perforar 200 nuevos pozos de petróleo mediante la creación de un modelo predictivo y la evaluación de diferentes regiones basadas en su rentabilidad y riesgo.

## Etapas del Proyecto

### Importación y Exploración de Datos

- Lectura de los datos geológicos de tres regiones distintas.
- Exploración inicial y análisis de las características de los datos.

### Preprocesamiento de Datos

- Limpieza y transformación de los datos.
- Dividir los datos en conjuntos de entrenamiento y prueba.
- Normalización y escalamiento de datos.

### Entrenamiento del Modelo

- Uso de regresión lineal para predecir el volumen de reservas.
- Evaluación del modelo utilizando validación cruzada.
- Selección de los 200 pozos con mayores predicciones de reservas en cada región.

### Evaluación de Beneficios

- Cálculo del beneficio esperado basado en las predicciones.
- Comparación de beneficios entre las tres regiones.

### Análisis de Riesgos

- Uso de la técnica de bootstrapping para evaluar la distribución de beneficios.
- Cálculo del intervalo de confianza y el riesgo de pérdidas.

## Conclusiones y Recomendaciones

### Conclusiones Generales

- Las predicciones del modelo de regresión lineal indican diferencias significativas en las reservas de petróleo entre las regiones.
- La región seleccionada ofrece el mayor beneficio esperado con el menor riesgo de pérdida.

### Recomendaciones

- Se recomienda perforar en la región con el mayor margen de beneficio y menor riesgo, optimizando así los recursos y maximizando la rentabilidad.
- Considerar un análisis de sensibilidad para ajustar los parámetros del modelo y mejorar la precisión de las predicciones.

## Tecnologías Utilizadas

- **Python:** Lenguaje de programación principal.
- **Pandas:** Para la manipulación y análisis de datos.
- **NumPy:** Para cálculos numéricos.
- **Scikit-Learn:** Para la implementación de modelos de Machine Learning.
- **Matplotlib:** Para la visualización de datos.
- **SciPy:** Para análisis estadístico y bootstrapping.

## Habilidades Relacionadas a Ciencia de Datos Desarrolladas

- **Análisis Exploratorio de Datos (EDA):** Comprensión y visualización de datos geológicos.
- **Preprocesamiento de Datos:** Limpieza, normalización y división de datos.
- **Modelado Predictivo:** Implementación y evaluación de modelos de regresión lineal.
- **Evaluación de Riesgos:** Análisis de riesgos mediante técnicas de bootstrapping.
- **Visualización de Datos:** Creación de gráficos y visualizaciones para interpretar resultados.
- **Toma de Decisiones Basada en Datos:** Propuestas estratégicas basadas en análisis cuantitativos.

## Archivos del Proyecto

- **Notebook:** Contiene el código y análisis completo del proyecto.
- **Dataset:** Archivos CSV con datos de exploración geológica utilizados en el análisis.
- **Resultados:** Documentación de conclusiones y recomendaciones basadas en el análisis realizado.

## Contenido de Archivos

- **.gitignore:** Plantilla de archivos a ignorar.
- **README.md:** Descripción del proyecto, herramientas utilizadas y detalles del análisis.
- **requirements.txt:** Requisitos mínimos de librerías necesarias para ejecutar el código.


## [Sprint 11 Algebra lineal](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S11%20Algebra%20lineal/P11_Algebra%20lineal.ipynb)

### Proyecto de Predicción de Consumo de Energía

#### Descripción del Proyecto
En este proyecto, se ha desarrollado un modelo de predicción del consumo de energía utilizando datos reales y un conjunto de algoritmos de aprendizaje automático. El objetivo principal es predecir el consumo de energía eléctrica en diferentes sectores y periodos de tiempo, para ayudar en la planificación y optimización del uso de recursos energéticos.

#### Objetivo
Desarrollar un modelo preciso de predicción del consumo de energía eléctrica que permita a las empresas y a las autoridades gubernamentales tomar decisiones informadas sobre la gestión y distribución de recursos energéticos.

#### Etapas del Proyecto
1. **Recopilación de Datos**: Obtención de datos históricos de consumo de energía de diversas fuentes.
2. **Preprocesamiento de Datos**: Limpieza y preparación de los datos para su análisis, incluyendo la normalización y el manejo de valores faltantes.
3. **Exploración de Datos**: Análisis exploratorio de datos para identificar patrones y tendencias.
4. **Selección de Modelos**: Evaluación de varios algoritmos de aprendizaje automático para determinar el más adecuado para la predicción del consumo de energía.
5. **Entrenamiento del Modelo**: Entrenamiento del modelo seleccionado con los datos históricos.
6. **Evaluación del Modelo**: Validación del modelo mediante métricas de evaluación como el error cuadrático medio (RMSE) y el coeficiente de determinación (R²).
7. **Implementación**: Despliegue del modelo en un entorno de producción para realizar predicciones en tiempo real.
8. **Monitoreo y Mantenimiento**: Seguimiento del rendimiento del modelo y ajustes necesarios para mantener su precisión a lo largo del tiempo.

#### Tecnologías Utilizadas
- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

#### Habilidades Relacionadas a Ciencia de Datos Desarrolladas
- **Manejo y preprocesamiento de datos**: Limpieza, normalización y transformación de conjuntos de datos.
- **Análisis exploratorio de datos**: Visualización y análisis de patrones en los datos.
- **Modelado predictivo**: Aplicación de algoritmos de aprendizaje automático para la predicción de series temporales.
- **Validación de modelos**: Evaluación de la precisión y efectividad de modelos predictivos.
- **Implementación en producción**: Despliegue de modelos en entornos de producción para realizar predicciones en tiempo real.
- **Monitoreo y mantenimiento de modelos**: Seguimiento y ajuste de modelos para asegurar su rendimiento continuo.

#### Conclusiones
El proyecto ha demostrado la viabilidad de utilizar algoritmos de aprendizaje automático para predecir el consumo de energía con un alto grado de precisión. Las predicciones generadas por el modelo pueden ayudar significativamente en la planificación y optimización del uso de recursos energéticos, contribuyendo a la eficiencia y sostenibilidad energética.



## [Sprint 12 Métodos numéricos (Predicción de Precios de Coches)](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S12%20M%C3%A9todos%20Num%C3%A9ricos/P12_Metodos%20num%C3%A9ricos.ipynb)

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


## [Sprint 13: Predicción de Demanda de Taxis en la Ciudad de Nueva York](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S13%20Series%20temporales/P13_Series%20temporales.ipynb)

### Descripción del Proyecto
Este sprint se centra en la predicción de la demanda de taxis en la Ciudad de Nueva York utilizando modelos de Machine Learning. Se desarrolla un modelo para prever el número de pedidos de taxis en la próxima hora, empleando datos históricos y diferentes técnicas de modelado para cumplir con una métrica de RECM en el conjunto de prueba.

### Objetivo
Construir un modelo predictivo que pueda predecir la demanda de taxis con alta precisión, cumpliendo con una métrica de RECM no superior a 48 en el conjunto de prueba.

### Etapas del Proyecto
- **Descarga y Preprocesamiento de Datos**: Obtención y preparación de datos históricos de pedidos de taxis.
- **Entrenamiento y Prueba de Modelos**: Implementación y evaluación de diferentes modelos de Machine Learning como LightGBM y CatBoost.
- **Optimización de Hiperparámetros**: Ajuste de hiperparámetros para mejorar el rendimiento del modelo.
- **Evaluación de Modelos**: Medición del rendimiento del modelo utilizando la métrica de RECM y otras métricas relevantes.

### Herramientas Tecnológicas Implementadas
- **LightGBM**: Modelado eficiente para datos grandes y complejos.
- **CatBoost**: Modelo basado en Gradient Boosting con manejo efectivo de variables categóricas.
- **Python Libraries**: Utilización de bibliotecas para preprocesamiento y modelado.

### Habilidades Relacionadas a Ciencia de Datos Desarrolladas
- **Modelado Predictivo**: Implementación de técnicas avanzadas para predicción de demanda.
- **Optimización de Modelos**: Ajuste y evaluación de modelos para cumplir con métricas específicas.
- **Análisis de Datos**: Preprocesamiento y análisis de grandes volúmenes de datos históricos.


## [Sprint 14 Aprendizaje automático para textos : Modelos de Clasificación de reseñas de películas en IMDB](https://github.com/OmiHdz/Data-Science-Portafolio/blob/6526e8fe69ec012a8155fcb6195eb5e152f27900/S14%20Aprendizaje%20autom%C3%A1tico%20para%20textos/P14_Aprendizaje%20autom%C3%A1tico%20para%20textos.ipynb)

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
