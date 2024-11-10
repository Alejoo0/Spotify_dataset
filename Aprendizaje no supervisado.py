import pandas as pd
import numpy as np

archivo   = 'Spotify_dataset/Spotify_Most_Streamed_Songs_Graficar.csv'
df = pd.read_csv(archivo, encoding='latin-1', delimiter=',')

# Limpiar nombres de columnas (eliminar espacios y paréntesis)
df.columns = [col.replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]

# Convertir datos categóricos a texto adecuado para WEKA
df['key'] = df['key'].astype('category')
df['mode'] = df['mode'].astype('category')
df['estacion'] = df['estacion'].astype('category')

# Eliminar saltos de línea y comas en las celdas de texto
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].replace({r'\n': ' ', r',': ' '}, regex=True)

# Guardar en formato CSV sin índice y asegurando el formato adecuado
df.to_csv('Spotify_dataset/Spotify_Most_Streamed_Songs_WEKA.csv', index=False, quoting=1)  # quoting=1 para rodear texto con comillas