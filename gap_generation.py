import numpy as np


def generar_huecos_por_dias(df, columna_fecha, columna_valor, porcentaje_dias, tamanio_hueco, cantidad_huecos, random_state=None):
    """
    Versión corregida que garantiza huecos del tamaño solicitado.
    """
    if random_state is None:
        random_state = 42  # Valor por defecto
    
    df_con_huecos = df.copy()
    dias_unicos = df[columna_fecha].dt.date.unique()  # Usamos date en lugar de dayofyear para mayor precisión
    num_dias_huecos = max(1, int(len(dias_unicos) * porcentaje_dias))
    
    # Seleccionar días aleatorios
    dias_huecos = np.random.choice(dias_unicos, size=num_dias_huecos, replace=False)
    
    huecos_indices = []
    
    for dia in dias_huecos:
        datos_dia = df_con_huecos[df_con_huecos[columna_fecha].dt.date == dia]
        indices_dia = datos_dia.index.to_numpy()
        
        # Si el día no tiene suficientes registros, saltarlo
        if len(indices_dia) < tamanio_hueco:
            continue
            
        # Intentar generar la cantidad solicitada de huecos
        huecos_generados = 0
        intentos = 0
        max_intentos = 10 * cantidad_huecos  # Límite para evitar bucles infinitos
        
        while huecos_generados < cantidad_huecos and intentos < max_intentos:
            intentos += 1
            
            # Seleccionar posición inicial aleatoria
            inicio = np.random.randint(0, len(indices_dia) - tamanio_hueco + 1)
            fin = inicio + tamanio_hueco
            indices_hueco = indices_dia[inicio:fin]
            
            # Verificar que no haya NaNs existentes en esta región
            if not df_con_huecos.loc[indices_hueco, columna_valor].isna().any():
                df_con_huecos.loc[indices_hueco, columna_valor] = np.nan
                huecos_indices.extend(indices_hueco)
                huecos_generados += 1
    
    return df_con_huecos, huecos_indices
