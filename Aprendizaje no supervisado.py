import pandas as pd
import numpy as np

archivo   = 'Spotify_Most_Streamed_Songs_Graficar.csv'
dataframe = pd.read_csv(archivo, encoding='latin-1', delimiter=',')