import numpy as np
import random
import pandas as pd
import sys

# --- Ajustar esta ruta a tu entorno ---
ruta_directorio = '/home/cony/Documentos/GEERS/codigos_pc'
sys.path.append(ruta_directorio)
import Metrics as m

from models import procesar_configuracion_huecos


# Evaluar y mostrar resultados solo en los huecos
def metricas(y_true, y_pred):
    MBE = m.mbe(y_true, y_pred)
    rMBE = m.rmbe(y_true, y_pred)
    RMSE = m.rmsd(y_true, y_pred)
    rRMSE = m.rrmsd(y_true, y_pred)
    MAE = m.mae(y_true, y_pred)
    rMAE = m.rmae(y_true, y_pred)
    return rMBE, rRMSE, rMAE


if __name__ == "__main__":

    # Configurar todas las semillas posibles
    np.random.seed(46)
    random.seed(46)

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
        # Agregar más configuraciones según sea necesario
    ]
    
    # Procesar todas las configuraciones
    mejor_semilla = 46
    resultados_totales = []
    for config in configuraciones:
        print(f"Procesando configuración: {config}")
        res = procesar_configuracion_huecos(df, config, guardar_modelos=True, random_state=mejor_semilla)
        resultados_totales.append(res)
    
    # Combinar todos los resultados
    df_resultados = pd.concat(resultados_totales, ignore_index=True)
    df_resultados.to_csv('df_resultados_gridsearch_3_4.csv', index=False)
    
    # Calcular métricas para cada configuración
    for config in configuraciones:
        config_str = str(config)
        mask = df_resultados['configuracion'] == config_str
        datos_config = df_resultados[mask]
        
        print(f"\nMétricas para configuración {config_str}:")
        print("Cams:", metricas(datos_config['valor_real'], datos_config['valor_GHIcams']))
        print("ERA:", metricas(datos_config['valor_real'], datos_config['valor_GHIera']))
        
        # Métricas para cada SLR
        for var in ['GHIcamscc', 'GHIcams', 'GHIeracc', 'GHIera', 'argp2']:
            print(f"SLR ({var}):", metricas(datos_config['valor_real'], datos_config[f'prediccion_SLR_{var}']))
        
        print("MLR:", metricas(datos_config['valor_real'], datos_config['prediccion_MLR']))
        print("MLP:", metricas(datos_config['valor_real'], datos_config['prediccion_MLP']))
