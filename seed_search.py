import numpy as np
import random
import pandas as pd
from sklearn.model_selection import KFold
import sys

# --- Ajustar esta ruta a tu entorno ---
ruta_directorio = '/home/cony/Documentos/GEERS/codigos_pc'
sys.path.append(ruta_directorio)
import Metrics as m

from gap_generation import generar_huecos_por_dias


# Evaluar y mostrar resultados solo en los huecos
def metricas(y_true, y_pred):
    MBE = m.mbe(y_true, y_pred)
    rMBE = m.rmbe(y_true, y_pred)
    RMSE = m.rmsd(y_true, y_pred)
    rRMSE = m.rrmsd(y_true, y_pred)
    MAE = m.mae(y_true, y_pred)
    rMAE = m.rmae(y_true, y_pred)
    return rMBE, rRMSE, rMAE


def encontrar_mejor_semilla(df, config_huecos, vars_regresion=None, 
                           vars_slr=None, semillas_prueba=range(42, 52), 
                           metrica_evaluacion='rRMSD'):
    """
    Encuentra la semilla que produce los mejores resultados según la métrica especificada
    
    Args:
        df: DataFrame con los datos
        config_huecos: Configuración de huecos a usar
        vars_regresion: Variables para regresión múltiple
        vars_slr: Variables para regresiones simples
        semillas_prueba: Rango de semillas a evaluar
        metrica_evaluacion: Métrica para comparar ('RMSE', 'MAE', 'rRMSE')
    
    Returns:
        mejor_semilla: Semilla que produjo mejores resultados
        metricas_todas: DataFrame con métricas de todas las semillas
        mejores_resultados: DataFrame con los resultados de la mejor semilla
    """
    
    # Configuración por defecto
    if vars_regresion is None:
        vars_regresion = ['GHIcamscc', 'TOA', 'argp2', 'GHIeracc', 'GHIera','GHIcams', 'sza', 'mak', 'ie']
    
    metricas_todas = {'semilla': [], 'rMSE_MLP': [], 'rRMSD_MLP': [], 'rMAE_MLP': []}
    mejor_metrica = float('inf')
    mejor_semilla = None
    mejores_resultados = None
    
    for semilla in semillas_prueba:
        print(f"\nEvaluando semilla: {semilla}")
        
        # Fijar semillas
        np.random.seed(semilla)
        random.seed(semilla)
        
        # Procesar con la semilla actual (sin guardar modelos)
        resultados = procesar_configuracion_huecos(
            df, config_huecos, vars_regresion, vars_slr, 
            guardar_modelos=False, random_state=semilla
        )
        
        # Calcular métricas
        y_true = resultados['valor_real']
        y_pred = resultados['prediccion_MLP']
        
        rmse = m.rmsd(y_true, y_pred)
        rrmsd = m.rrmsd(y_true, y_pred)
        rmae = m.mae(y_true, y_pred)
        
        # Guardar métricas
        metricas_todas['semilla'].append(semilla)
        metricas_todas['rMSE_MLP'].append(rmse)
        metricas_todas['rRMSD_MLP'].append(rrmsd)
        metricas_todas['rMAE_MLP'].append(rmae)
        
        # Determinar si es la mejor según la métrica seleccionada
        metrica_actual = rmse if metrica_evaluacion == 'rMSE' else rrmsd if metrica_evaluacion == 'rRMSD' else rmae
        print(f"Semilla {semilla} con {metrica_evaluacion}: {metrica_actual:.4f}")
        
        if metrica_actual < mejor_metrica:
            mejor_metrica = metrica_actual
            mejor_semilla = semilla
            mejores_resultados = resultados
            print(f"¡Nueva mejor semilla! {semilla} con {metrica_evaluacion}: {metrica_actual:.4f}")
    
    return mejor_semilla, pd.DataFrame(metricas_todas), mejores_resultados


def procesar_configuracion_huecos(df, config_huecos, vars_regresion=None, vars_slr=None, 
                                 guardar_modelos=False, random_state=None):
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV
    import joblib

    # Fijar semillas al inicio de la función
    if random_state is None:
        random_state = 42  # Valor por defecto

    np.random.seed(random_state)
    random.seed(random_state)
    # Configuración por defecto para variables
    if vars_regresion is None:
        vars_regresion = ['GHIcamscc', 'TOA', 'argp2', 'GHIeracc', 'GHIera','GHIcams', 'sza', 'mak', 'ie']
    
    # Variables para las SLR
    vars_slr = ['GHIcamscc', 'GHIcams', 'GHIeracc', 'GHIera', 'argp2']
    
    # 1. Generar huecos (sin cambios)
    d_huecos, huecos_indices = generar_huecos_por_dias(
        df, 'date', 'ghi',
        porcentaje_dias=config_huecos['porcentaje_dias'],
        tamanio_hueco=config_huecos['tamanio_hueco'],
        cantidad_huecos=config_huecos['cantidad_huecos'],
        random_state=random_state
    )
    
 # 2. Preparar datos para modelos
    X_huecos = d_huecos.drop(columns=['ghi'])
    y = df['ghi'].squeeze()
    
    # Escalado para MLP
    scaler = StandardScaler()
    X_huecos_scaled = scaler.fit_transform(X_huecos.drop(columns=['date']))
    

    # 2. Entrenar modelos SLR adicionales (solo este bloque cambia)
    modelos_slr = {}
    for var in vars_slr:
        modelo = LinearRegression()
        modelo.fit(df[[var]], y)
        modelos_slr[var] = modelo
    
   
    # 3. Entrenar modelos
    
    # Regresión Lineal Múltiple
    modelo_mlr = LinearRegression()
    modelo_mlr.fit(df[vars_regresion], y)
    
    # MLP con GridSearch
    modelo_mlp = MLPRegressor(max_iter=1000, early_stopping=True, activation='relu', random_state=random_state, solver='adam')
    n_features = len(vars_regresion)
    param_grid = {
        'hidden_layer_sizes': [
            (n_features,), (n_features * 2,), 
            (n_features, n_features), (n_features, n_features, n_features),
            (n_features, n_features * 2), (n_features, n_features, n_features * 2)
        ],
        'alpha': [0.0001, 0.001, 0.01]
    }
    # Crear un objeto KFold con semilla
    cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(modelo_mlp, param_grid, cv=cv, n_jobs=-1, verbose=0)
    grid_search.fit(X_huecos_scaled, y)
    
        # === EXTRACCIÓN DE LA MEJOR CONFIGURACIÓN ===
    mejor_config = grid_search.best_params_
    print(f"\n🔍 Mejor configuración MLP para {config_huecos}:")
    print(f"• Capas ocultas: {mejor_config['hidden_layer_sizes']}")
    print(f"• Alpha (regularización L2): {mejor_config['alpha']}")
    print(f"• Score (R²) en validación: {grid_search.best_score_:.4f}")

    mejor_modelo = grid_search.best_estimator_

    # 4. Realizar predicciones
    resultados_dict = {
        'fecha_hueco': df.loc[huecos_indices, 'date'],
        'valor_real': df.loc[huecos_indices, 'ghi'],
        'valor_GHIcams': df.loc[huecos_indices, 'GHIcams'],
        'valor_GHIera': df.loc[huecos_indices, 'GHIera'],
        'prediccion_MLR': modelo_mlr.predict(d_huecos.loc[huecos_indices, vars_regresion]),
        'prediccion_MLP': mejor_modelo.predict(X_huecos_scaled[huecos_indices]),
        'configuracion': str(config_huecos)
    }
    
    # Agregar predicciones SLR
    for var in vars_slr:
        resultados_dict[f'prediccion_SLR_{var}'] = modelos_slr[var].predict(d_huecos.loc[huecos_indices, [var]]) 
    
    resultados = pd.DataFrame(resultados_dict).sort_values('fecha_hueco').reset_index(drop=True)

    # 5. Guardar modelos
    if guardar_modelos:
        config_str = f"h{config_huecos['tamanio_hueco']}_p{int(config_huecos['porcentaje_dias']*100)}_n{config_huecos['cantidad_huecos']}"
        
        # Guardar modelos SLR
        for var in vars_slr:
            joblib.dump(modelos_slr[var], f'modelo_SLR_{var}_{config_str}.pkl')
        
        # Guardar otros modelos
        joblib.dump(modelo_mlr, f'modelo_MLR_{config_str}.pkl')
        joblib.dump(mejor_modelo, f'modelo_MLP_{config_str}.pkl')
        joblib.dump(scaler, f'scaler_{config_str}.pkl')
    
    return resultados


if __name__ == "__main__":
    # Cargar datos
    ruta = r'D:\GEERS\env_trabajo\CursoRadiacion2024\trabajoFinal\measured.csv'
    df = pd.read_csv(ruta) 
    df['date'] = pd.to_datetime(df['date'])
    df = df[df.sza < 85]
    
    # Definir configuraciones a probar
    configuraciones = [
        {'tamanio_hueco': 1, 'porcentaje_dias': 0.1, 'cantidad_huecos': 12},
        {'tamanio_hueco': 4, 'porcentaje_dias': 0.1, 'cantidad_huecos': 3},
        {'tamanio_hueco': 1, 'porcentaje_dias': 0.3, 'cantidad_huecos': 12},
        {'tamanio_hueco': 4, 'porcentaje_dias': 0.3, 'cantidad_huecos': 3},
        {'tamanio_hueco': 1, 'porcentaje_dias': 0.5, 'cantidad_huecos': 12},
        {'tamanio_hueco': 4, 'porcentaje_dias': 0.5, 'cantidad_huecos': 3},
        # ... otras configuraciones
    ] 
    
    # Diccionario para guardar resultados
    resultados_finales = {
        'configuracion': [],
        'mejor_semilla': [],
        'metricas': [],
        'resultados': []
    }
    
    # Procesar configuraciones solo para evaluación de semillas
    for config in configuraciones:
        print(f"\n{'='*50}\nEvaluando configuración: {config}\n{'='*50}")
        
        # Encontrar mejor semilla y métricas
        mejor_semilla, metricas_semillas, resultados = encontrar_mejor_semilla(df, config)
        
        # Guardar resultados
        resultados_finales['configuracion'].append(str(config))
        resultados_finales['mejor_semilla'].append(mejor_semilla)
        resultados_finales['metricas'].append(metricas_semillas)
        resultados_finales['resultados'].append(resultados)
    
    # Guardar resumen final
    pd.DataFrame({
        'configuracion': resultados_finales['configuracion'],
        'mejor_semilla': resultados_finales['mejor_semilla']
    }).to_csv('resumen_semillas_optimas.csv', index=False)
