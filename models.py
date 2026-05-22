import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
import joblib

from gap_generation import generar_huecos_por_dias


def procesar_configuracion_huecos(df, config_huecos, vars_regresion=None, vars_slr=None, guardar_modelos=False, random_state=None):
    """
    Procesa una configuración de huecos específica y devuelve las predicciones.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos completos
        config_huecos (dict): Configuración de los huecos a generar
        vars_regresion (list): Variables para la regresión múltiple
        vars_slr (list): Variables para regresiones lineales simples
        guardar_modelos (bool): Si True, guarda los modelos en disco
        
    Returns:
        pd.DataFrame: Resultados de las predicciones para los huecos generados
    """
    if random_state is None:
        random_state = 42
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
        random_state=config_huecos.get('random_state', random_state)
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
