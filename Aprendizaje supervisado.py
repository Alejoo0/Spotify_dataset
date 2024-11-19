from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cargar el dataset y preparar los datos
archivo = 'Spotify_dataset/Spotify_Most_Streamed_Songs_ANALISIS.csv'
df = pd.read_csv(archivo, encoding='latin-1', delimiter=',')

# Eliminar columnas irrelevantes
columnas_eliminar = ['track_name', 'artist(s)_name', 'released_year', 'released_month',
                     'released_day', 'in_spotify_charts', 'streams', 'cover_url']
df = df.drop(columns=columnas_eliminar)

df['mode'] = df['mode'].replace({'Minor': 0, 'Major': 1})
df['key'] = df['key'].replace({
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
})
df['estacion'] = df['estacion'].replace({
    'Primavera': 0, 'Otoño': 1, 'Verano': 2, 'Invierno': 3
})
df = df.dropna()


# Crear rangos automáticos para la variable objetivo
num_rangos = 4
etiquetas = ['Baja', 'Media', 'Alta', 'Muy Alta']
df['in_spotify_playlists_rangos'] = pd.cut(
    df['in_spotify_playlists'], 
    bins=num_rangos, 
    labels=etiquetas, 
    right=False
)

atributos = ['artist_count','bpm','mode','danceability_%','valence_%','energy_%','acousticness_%',
             'instrumentalness_%','liveness_%','speechiness_%']
X = df[atributos]
y = df['in_spotify_playlists_rangos']

# Convertir las características a tipo float
X = np.array(X, dtype='float64')
y_encoded, y_mapping = pd.factorize(y)

# Validación cruzada con balanceo usando SMOTE
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # División estratificada para conservar la distribución de clases
accuracy_scores = []

for train_index, test_index in cv.split(X, y_encoded):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    # Aplicar SMOTE en cada partición de entrenamiento
    smote = SMOTE(random_state=42, k_neighbors=2)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # Entrenar el modelo
    modelo_clasificacion = DecisionTreeClassifier(max_depth=4, random_state=42)
    modelo_clasificacion.fit(X_train_bal, y_train_bal)
    
    # Evaluar en el conjunto de prueba
    y_pred_test = modelo_clasificacion.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    accuracy_scores.append(acc)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

 # Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred_test)
    
# Visualizar la matriz de confusión con un heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y_mapping.astype(str), yticklabels=y_mapping.astype(str))
plt.title("Matriz de Confusión")
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Imprimir reporte para cada partición (corrección aquí)
print("\nReporte de clasificación en una partición:")
print(classification_report(
    y_test, 
    y_pred_test, 
    target_names=y_mapping.astype(str), 
    labels=np.arange(len(y_mapping))  # Considerar todas las clases
))

# Resultados finales
print("\nPuntajes de precisión en validación cruzada:", accuracy_scores)
print("\nPrecisión promedio en validación cruzada:", np.mean(accuracy_scores))

from sklearn.tree import plot_tree
# Visualizar el árbol de decisión de manera ordenada
plt.figure(figsize=(28, 18))
plot_tree(
    modelo_clasificacion, 
    feature_names=atributos, 
    class_names=y_mapping.astype(str), 
    filled=True, 
    fontsize=6, 
    max_depth=4,  # Limitar la profundidad para mejorar la claridad
    rounded=True,  # Bordes redondeados para un estilo más limpio
    proportion=False  # Hace que los tamaños de los nodos estén proporcionales al número de muestras
)
plt.title("Árbol de Decisión (Última partición)")
plt.show()

