import pandas as pd
import numpy as np

archivo   = 'Spotify_dataset/Spotify_Most_Streamed_Songs_Graficar.csv'
df = pd.read_csv(archivo, encoding='latin-1', delimiter=',')

# Convertir datos categóricos a texto adecuado para WEKA
df['key'] = df['key'].astype('category')
df['mode'] = df['mode'].astype('category')
df['estacion'] = df['estacion'].astype('category')

#PARA ELIMINAR OUTLAYER 
# Eliminando las filas donde los streams son mayores a 3,500,000,000
df =df[df['streams'] <= 3500000000]

# Eliminar la columna 'artist_name'
df = df.drop(columns=['artist(s)_name'])
# Guardar en formato CSV sin índice y asegurando el formato adecuado

df.info()
df.to_csv('Spotify_dataset/Spotify_Most_Streamed_Songs_WEKA.csv', index=False)  # quoting=1 para rodear texto con comillas
