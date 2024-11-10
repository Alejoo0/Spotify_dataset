import pandas as pd
import numpy as np

archivo   = 'Spotify_dataset/Spotify_Most_Streamed_Songs_Graficar.csv'
dataframe = pd.read_csv(archivo, encoding='latin-1', delimiter=',')

# Convertir 'key' y 'mode' a tipo categórico
dataframe['key']  = dataframe['key'].astype('category')
dataframe['mode'] = dataframe['mode'].astype('category')

#PARA ELIMINAR OUTLAYER 
# Eliminando las filas donde los streams son mayores a 3,500,000,000
dataframe = dataframe[dataframe['streams'] <= 3500000000]

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

#HISTOGRAMA STREAMS
""""
# Crear una función para formatear los números en el eje X
def formatter(x, pos):
    return f'{int(x):,}'  # Formato con comas para miles

# Crear el histograma de streams
plt.figure(figsize=(16, 12))
plt.hist(dataframe['streams'], bins=120, color='orange', edgecolor='black')
plt.title('Distribución de Streams')
plt.xlabel('Streams')
plt.ylabel('Frecuencia')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(range(0, 4000000000, 250000000))

# Ajustar los límites en el eje X para que comience desde el mínimo valor de streams
plt.xlim(left=dataframe['streams'].min())

# Aplicar el formateador personalizado en el eje X
plt.gca().xaxis.set_major_formatter(FuncFormatter(formatter))

plt.show()
"""

# DENSIDAD BPM
""""
plt.figure(figsize=(8, 4))
sns.histplot(dataframe['bpm'], bins=60, color='green', kde=True)
plt.title('Distribución de BPM')
plt.xlabel('BPM')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.xlim(left=dataframe['bpm'].min())
plt.xticks(range(0, 211, 10))
plt.show()

"""
# DENSIDAD SPOTIFY PLAYLIST
""""
plt.figure(figsize=(16, 6))
sns.histplot(dataframe['in_spotify_playlists'], bins=260, palette="flare")
plt.title('Distribución de canciones en playlist')
plt.xlabel('playlist')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.xlim(left=dataframe['in_spotify_playlists'].min())
plt.xticks(range(0, 55001, 2500))
plt.show()
"""

# BARRAS APILADAS MODE Y KEYS
"""
plt.figure(figsize=(10, 6))
sns.countplot(data=dataframe, x='key', hue='mode', palette='Blues')
plt.title('Distribución de Notas por Modos (Mayor y Menor)')
plt.xlabel('Key')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.legend(title='Mode')
plt.grid(axis='y', alpha=0.75)
plt.show()
"""

# HISTOGRAMA PARA VARIABLES TIPO _%
"""
plt.figure(figsize=(10, 6))
sns.histplot(dataframe['speechiness_%'], bins=80, kde=True, color='purple')
plt.title('Distribución de speechiness_%')
plt.xlabel('speechiness_%')
plt.ylabel('Frecuencia')
plt.grid(axis='y', alpha=0.75)
plt.xlim(left=dataframe['speechiness_%'].min())
plt.xticks(range(0, 101, 5))
plt.show()
"""
# DIAGRAMA CANCIONES POR AÑO
"""
plt.figure(figsize=(12, 6))
sns.countplot(data=dataframe, x='released_year', palette="flare", edgecolor='black')
plt.title('Distribución de Canciones Lanzadas por Año')
plt.xlabel('Año de Lanzamiento')
plt.ylabel('Cantidad de Canciones')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.yticks(range(0, 381, 20))
#
plt.show()
"""

# DIAGRAMA CANCIONES POR ESTACIONES DEL AÑO 
"""
# Contar la cantidad de canciones lanzadas por estación
season_counts = dataframe['estacion'].value_counts()

# Definir colores para cada estación
colors = ['lightblue', 'lightgreen', 'salmon', 'gold']  # Colores para cada estación

# Crear un gráfico de barras con diferentes colores
plt.figure(figsize=(8, 6))
plt.bar(season_counts.index, season_counts.values, color=colors, edgecolor='black')
plt.title('Distribución de Canciones Lanzadas por Estación del Año')
plt.xlabel('Estación')
plt.ylabel('Cantidad de Canciones')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
plt.yticks(range(0, 281, 20))
plt.show()

"""

#MATRIZ DE CORRELACION
""""
# Seleccionar solo las columnas numéricas para la matriz de correlación
numerical_dataframe = dataframe.select_dtypes(include=['float64', 'int64'])

# Calcular la matriz de correlación
correlation_matrix = numerical_dataframe.corr()

# Crear un mapa de calor para visualizar la matriz de correlación
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación de Variables Numéricas')
plt.show()
"""

#CODO PARA SACAR NUMERO DE K PARA K-MEANS
""""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Selección de las columnas relevantes para el clustering
features = [
    'artist_count','released_year','in_spotify_playlists','bpm', 'danceability_%', 'energy_%', 'valence_%', 'acousticness_%',
    'instrumentalness_%', 'liveness_%', 'speechiness_%', 'streams'
]
X = dataframe[features]

# Estandarización de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método del Codo para determinar el número óptimo de clusters
inertia = []
K = range(1, 24)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)



import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reducción de dimensionalidad con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Crear gráfico de dispersión de los clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', marker='o', edgecolor='k')
plt.xlabel('PCA Componente 1')
plt.ylabel('PCA Componente 2')
plt.title('Visualización de Clusters con PCA')
plt.colorbar(label='Cluster')
plt.show()
"""