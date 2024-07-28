## Sprint 4 Análisis estadístico de datos (Modelo de recomendación de planes de tarifas telefónicas)

# Proyecto de Análisis de Datos: Análisis Estadístico de Planes Telefónicos

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
