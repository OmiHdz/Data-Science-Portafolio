## Sprint 3 Data Wrangling
# Proyecto de Análisis de Datos: Comportamiento de Compras en una Plataforma de E-commerce

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
