# Reconstrucción de Series de Irradiancia Global Horizontal (GHI) mediante Machine Learning

Este repositorio contiene la implementación algorítmica desarrollada para la imputación de datos faltantes (**gap-filling**) en series temporales de Irradiancia Global Horizontal (GHI). El proyecto se centra en el uso de técnicas de **Machine Learning** y la integración de datos satelitales y de reanálisis para asegurar la integridad de las bases de datos radiométricas [5, 6].

## Descripción del Proyecto
La falta de registros por fallas de instrumentación o mantenimiento es un desafío crítico en el análisis del recurso solar [7]. Este código implementa y compara tres estrategias de imputación:
*   **Regresión Lineal Simple (SLR)**.
*   **Regresión Lineal Múltiple (MLR)**.
*   **Perceptrón Multicapa (MLP)**: Arquitectura que demostró el mejor desempeño, con un rRMSD entre 18% y 21% [2, 5].

### Caso de Estudio
La metodología fue validada utilizando una serie de datos con resolución de **15 minutos** de la localidad de **El Rosal, Salta, Argentina** (3355 m.s.n.m) [8, 9].

## Fuentes de Datos Integradas
Para mejorar la precisión de las predicciones, el algoritmo permite integrar variables de [8, 10]:
*   **CAMS** (Copernicus Atmosphere Monitoring Service).
*   **ERA5** (Reanálisis del ECMWF).
*   **ARGP2**: Modelo de cielo claro ajustado para el noroeste argentino.

## Estructura del Repositorio
## Estructura

```
├── gap_generation.py   # Generación de huecos sintéticos
├── models.py           # Entrenamiento de modelos (SLR, MLR, MLP)
├── run_gridsearch.py   # Pipeline principal → genera df_resultados_gridsearch_3_4.csv
└── seed_search.py      # Búsqueda de semilla óptima → genera resumen_semillas_optimas.csv
```

## Uso
**1. Ajustar la ruta del módulo Metrics en cada script:**
```python
ruta_directorio = '/ruta/a/tus/codigos_pc'
```
**2. Ajustar la ruta del CSV:**
```python
ruta = '/ruta/a/measured.csv'
```
**3. Correr:**
```bash
# Buscar semilla óptima primero (opcional)
python seed_search.py

# Pipeline principal
python run_gridsearch.py
```
## Dependencias

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```
## Referencia de Publicación
Este trabajo técnico y científico se encuentra documentado en:
*   **López Ruiz, C. B.**, Ledesma, R. D., Salazar, G. A., y Galdiño, J. (2025). *"Reconstrucción de series de irradiancia global horizontal con huecos sintéticos mediante modelos de machine learning y datos satelitales. Caso de estudio: El Rosal, Salta"*. **Avances en Energías Renovables y Medio Ambiente - AVERMA**, Vol. 29, pp. 484-499.
*   Presentado en la **XLVII Reunión de Trabajo de ASADES (2025)**.


## Autora y Contacto
**Lic. Constanza B. López Ruiz**  
Profesional Asistente (CPA) - **INENCO** (CONICET - UNSa)  
conyblopezruiz@gmail.com.

