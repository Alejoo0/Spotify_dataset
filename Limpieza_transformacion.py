import pandas as pd
import numpy as np

archivo   = 'Spotify_dataset/Spotify Most Streamed Songs.csv'
dataframe = pd.read_csv(archivo, encoding='latin-1', delimiter=',')

# Eliminar Columnas(Son otras plataformas)
columnas_eliminar = ['in_apple_playlists','in_apple_charts','in_deezer_playlists','in_deezer_charts',
                     'in_shazam_charts']
dataframe = dataframe.drop(columns=columnas_eliminar)

# Eliminar la fila número 574 (Contiene error en columna streams)
dataframe = dataframe.drop(574)

# Convertir valores object a tipo int (numero de streams)
dataframe['streams'] = dataframe['streams'].astype('int64')


# Eliminar filas donde el valor en la columna 'key' es nulo
dataframe = dataframe.dropna(subset=['key'])


# Función para determinar la estación considerando el mes y el día
def obtener_estacion(mes, dia):
    if (mes == 12 and dia >= 21) or mes in [1, 2] or (mes == 3 and dia <= 20):
        return 'Invierno'
    elif (mes == 3 and dia >= 21) or mes in [4, 5] or (mes == 6 and dia <= 20):
        return 'Primavera'
    elif (mes == 6 and dia >= 21) or mes in [7, 8] or (mes == 9 and dia <= 22):
        return 'Verano'
    elif (mes == 9 and dia >= 23) or mes in [10, 11] or (mes == 12 and dia <= 20):
        return 'Otoño'

# Crear la nueva columna 'estacion' aplicando la función
dataframe['estacion'] = dataframe.apply(lambda row: obtener_estacion(row['released_month'], row['released_day']), axis=1)

# Codificar estación del año en valores numéricos (1 a 4)
#estaciones_cod = {'Invierno': 1, 'Primavera': 2, 'Verano': 3, 'Otoño': 4}
#dataframe['estacion_numerica'] = dataframe['estacion'].map(estaciones_cod)
#mode_cod = {'Major':1,'Minor':0}
#dataframe['mode'] = dataframe['mode'].map(mode_cod)

# Convertir 'key' y 'mode' a tipo categórico
dataframe['key']  = dataframe['key'].astype('category')
dataframe['mode'] = dataframe['mode'].astype('category')

dataframe = dataframe[dataframe['streams'] <= 3500000000]

# Guardar el DataFrame en un nuevo archivo CSV
nuevo_archivo = 'Spotify_dataset/Spotify_Most_Streamed_Songs_ANALISIS.csv'
dataframe.to_csv(nuevo_archivo, index=False, encoding='latin-1')