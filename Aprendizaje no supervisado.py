import pandas as pd
import numpy as np

archivo   = 'Spotify_dataset/Spotify_Most_Streamed_Songs_ANALISIS.csv'
df = pd.read_csv(archivo, encoding='latin-1', delimiter=',')

# Convertir datos categóricos a texto adecuado para WEKA
df['key'] = df['key'].astype('category')
df['mode'] = df['mode'].astype('category')
df['estacion'] = df['estacion'].astype('category')

#PARA ELIMINAR OUTLAYER 
# Eliminando las filas donde los streams son mayores a 3,500,000,000
df =df[df['streams'] <= 3500000000]

""""
# Función para obtener el cuartil superior de streams por temporada, ordenado de manera descendente
def cuartil_superior_por_temporada(df, temporada_col="estacion", streams_col="streams", cuartil=0.75):
    temporadas = df[temporada_col].unique()
    resultado = {}
    
    for temporada in temporadas:
        # Filtrar por temporada
        df_temporada = df[df[temporada_col] == temporada]
        
        # Calcular el valor del cuartil superior
        limite_cuartil = df_temporada[streams_col].quantile(cuartil)
        
        # Filtrar canciones en el cuartil superior de streams
        df_cuartil_superior = df_temporada[df_temporada[streams_col] >= limite_cuartil]
        
        # Ordenar de manera descendente por streams
        df_cuartil_superior = df_cuartil_superior.sort_values(by=streams_col, ascending=False)
        
        # Guardar el resultado en un diccionario
        resultado[temporada] = df_cuartil_superior
    
    return resultado

# Obtener el cuartil superior de streams por temporada
cuartil_superior = cuartil_superior_por_temporada(df)
# Ajustar el ancho máximo de las columnas
pd.set_option('display.max_colwidth', None)

# Mostrar los resultados
for temporada, df_cuartil in cuartil_superior.items():
    print(f"\nCanciones en el cuartil superior de streams para {temporada}:")
    print(df_cuartil[["track_name", "streams", "artist(s)_name","cover_url"]])

"""

# Filtrar las canciones para cada estación

# Filtrar las canciones para cada estación
"""
# Invierno
df_invierno = df[
    (df['estacion'] == 'Invierno') &  # Filtrar por estación
    (df['danceability_%'] >= 60) &
    (df['valence_%'] >= 40) &
    (df['instrumentalness_%'] <= 2) &
    (df['energy_%'] >= 60) &
    (df['liveness_%'] >= 15)
]

pd.set_option('display.max_colwidth', None)
# Imprimir las canciones de cada estación
print("Canciones para Invierno:")
print(df_invierno[['track_name', 'streams', 'artist(s)_name', 'cover_url']])

"""