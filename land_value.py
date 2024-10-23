import streamlit as st

st.title("Prédiction de valeurs foncières")

import streamlit.components.v1 as components
'''
## Préparation des données
'''

"""
Ce projet consiste à analyser et prédire les valeurs foncières en France métropolitaine.\n
Nous utiliserons un dataset contenant des informations sur les transactions immobilières en France en 2023. Nous allons commencer par charger et explorer les données, puis nous allons entraîner des modèles de machine learning pour prédire les valeurs foncières.\n
Pour gagner du temps, seulement 10% des données seront chargées.\n
Source des données: """ 
st.page_link("https://www.data.gouv.fr/fr/datasets/r/78348f03-a11c-4a6b-b8db-2acf4fee81b1", label="data.gouv.fr")
""" Voici les données que nous garderons et étudierons dans le dataset:
"""

import pandas as pd
import numpy as np

# Load the dataset
url = "https://www.data.gouv.fr/fr/datasets/r/78348f03-a11c-4a6b-b8db-2acf4fee81b1"
dtypes = {
    'Date mutation': 'str',
    'Valeur fonciere': 'float',
    'Code departement': 'str',
    'Type local': 'str',
    'Surface reelle bati': 'float',
    'Nombre pieces principales': 'Int64',
    'Surface terrain': 'float',
}
# Create the DataFrame
df = pd.DataFrame(list(dtypes.items()), columns=["Nom de la colonne", "Type de données"])

st.table(df)

if (False):
    df = pd.read_csv(url, delimiter='|', decimal=",", na_values=[''], skiprows=lambda i: i>0 and (i%5!=0)) #version plus rapide pour machine learning
    #df = pd.read_csv(url, delimiter='|', decimal=",", na_values=[''])
    df['Type local'] = df['Type local'].fillna("NaN")
    #df['Surface reelle bati'] = df['Surface reelle bati'].fillna(df['Surface reelle bati'].mean())
    df['Nombre pieces principales'] = df['Nombre pieces principales'].fillna(df['Nombre pieces principales'].mean().round())
    #df['Surface terrain'] = df['Surface terrain'].fillna(df['Surface terrain'].mean())

    df = df.astype(dtypes)

    #garde uniquement le département 92
    #df = df[df['Code departement'] == '92']

    # Convertir 'Date mutation' en mois (float)
    df['Date mutation'] = pd.to_datetime(df['Date mutation'], format='%d/%m/%Y')
    df['Date mutation'] = df['Date mutation'].dt.month


    # Garder les colonnes pertinentes
    columns_to_keep = [
        'Date mutation', 'Valeur fonciere', 'Code departement', 'Type local',
        'Surface reelle bati', 'Nombre pieces principales', 'Surface terrain',
    ]
    df = df[columns_to_keep]


    df = df.dropna(subset=['Valeur fonciere', 'Surface reelle bati', 'Surface terrain'])
    # supression valeurs aberrantes
    df = df[df['Valeur fonciere']>10]
    df = df[df['Valeur fonciere']<5000000]
    df = df[df['Surface reelle bati']<100000]
    df = df[df['Surface terrain']<400000]
    df = df[df['Nombre pieces principales']<40]

    #reset the indexes
    df = df.reset_index(drop=True)

    df = df[df['Surface reelle bati'] > 0]

    df.to_pickle("datasets/data.pkl")  # where to save it, usually as a .pkl


df = pd.read_pickle("datasets/data.pkl")

"""## Partie Analyse Exploratoire Des Données"""

import seaborn as sns
import matplotlib.pyplot as plt

# Histogramme des valeurs foncières
fig = plt.figure(figsize=(10, 6))
sns.histplot(df['Valeur fonciere'], bins=50, kde=True)
plt.title('Distribution des Valeurs Foncières')
plt.xlabel('Valeur Fonciere')
plt.ylabel('Fréquence')
plt.show()
st.pyplot(fig)

# Boxplot des valeurs foncières par type de location
fig = plt.figure(figsize=(12, 8))
sns.boxplot(x='Type local', y='Valeur fonciere', data=df)
plt.title('Valeur Foncière par Type de Local')
plt.xticks(rotation=45)
plt.show()
st.pyplot(fig)

# Scatterplot Valeur foncière vs Surface réelle bâti
fig = plt.figure(figsize=(10, 6))
sns.scatterplot(x='Surface reelle bati', y='Valeur fonciere', data=df)
plt.title('Valeur Foncière vs Surface Réelle Bati')
plt.xlabel('Surface Réelle Bati (m²)')
plt.ylabel('Valeur Fonciere')
plt.ylim(0, 5000000)
plt.xlim(0, 10000)
plt.show()
st.pyplot(fig)

"""
On retiendra que:
 
-La grande majorité des ventes ont un montant inférieur à 1M d'euros\n
-Les maisons coûtent en général moins cher que les appartements. Cela s'explique par leur positionnement: les appartements se situent surtout en ville, tandis que la majorité des maisons se situe en campagne et banlieue. Les locaux commerciaux et industriels coûtent aussi plus cher par leur nature et leur taille.
"""

"""## Geomap"""

# Feature engineering

df['Prix_m2'] = df['Valeur fonciere'] / df['Surface reelle bati']

# Filtrer les valeurs aberrantes
df = df[df['Prix_m2'] < 30000]
df = df.reset_index(drop=True)

prix_m2_par_departement = df.groupby('Code departement')['Prix_m2'].mean().reset_index()
prix_m2_par_departement.columns = ['code', 'Prix_m2']

# Convertir le code département en string pour correspondre au geojson
prix_m2_par_departement['code'] = prix_m2_par_departement['code'].str.zfill(2)

# Formater les prix avec séparateurs de milliers et le symbole €
prix_m2_par_departement['Prix_m2_format'] = prix_m2_par_departement['Prix_m2'].apply(lambda x: f"{x:,.2f} €".replace(',', ' ').replace('.', ','))


# Geomap
import plotly.express as px
import geopandas as gpd
departements_geojson = gpd.read_file("departements.geojson")
# Créer la carte
fig = px.choropleth_mapbox(
    prix_m2_par_departement,
    geojson=departements_geojson,
    featureidkey="properties.code",
    locations='code',
    color='Prix_m2',
    color_continuous_scale="Viridis",
    mapbox_style="carto-positron",
    zoom=4,
    center={"lat": 46.603354, "lon": 1.888334},
    opacity=0.6,
    labels={'Prix_m2': 'Prix moyen par m²'},
    title='Prix moyen par m² par département',
    hover_data={'Prix_m2_format': True, 'Prix_m2': False}  # Utiliser la colonne formatée pour l'affichage
)
st.plotly_chart(fig)

"On remarque que les prix moyens par département les plus élevés se situent à Paris et dans le Sud-Est de la France."

"""## Histogramme"""

# Calcul de la moyenne des prix par département
mean_price_by_dept = df.groupby('Code departement')['Prix_m2'].mean()

# Création de la figure et des axes
fig, ax = plt.subplots(figsize=(30, 10))

# Tracé de la figure
mean_price_by_dept.plot(kind='bar', ax=ax)
ax.set_ylabel('Prix au m² moyen')
ax.set_title('Prix au m² par département')

# Affichage du graphique
plt.show()

# Utilisation de Streamlit pour afficher la figure
st.pyplot(fig)

"""## Matrice de corrélation"""

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Encoder les colonnes code departement et type local en utilisant OneHotEncoder
encoder = OneHotEncoder()
df.drop('Prix_m2', axis=1, inplace=True)
df_encoded = df.copy()

encoded = encoder.fit_transform(pd.DataFrame(df['Code departement'])).toarray()
codes = encoder.get_feature_names_out()
encoded_cols = pd.DataFrame(encoded, columns=codes)
df_encoded = pd.concat([df_encoded, encoded_cols], axis=1)
df_encoded = df_encoded.drop("Code departement", axis=1)

encoded = encoder.fit_transform(pd.DataFrame(df['Type local'])).toarray()
local = encoder.get_feature_names_out()
encoded_cols = pd.DataFrame(encoded, columns=local)
df_encoded = pd.concat([df_encoded, encoded_cols], axis=1)
df_encoded = df_encoded.drop("Type local", axis=1)

if (False):
    pd.to_pickle(codes, "datasets/codes.pkl")
    pd.to_pickle(local, "datasets/local.pkl")

"""A partir d'ici, les variables code département et type local sont encodées en utilisant OneHotEncoder. Nous allons maintenant standardiser les données et afficher la matrice de corrélation."""

#standardiser
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.set_output(transform='pandas')
df_scaled = scaler.fit_transform(df_encoded)

#Matrice de corrélation
#ne prendre que "Valeur fonciere", "Surface reelle bati", "Nombre pieces principales", "Surface terrain", "Date mutation"

correlation_matrix = df_scaled[['Valeur fonciere', 'Surface reelle bati', 'Nombre pieces principales', 'Surface terrain', 'Date mutation']].corr()
fig = plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de Corrélation')
plt.show()
st.pyplot(fig)
"""On voit sur la matrice de corrélation que la valeur foncière n'est pas fortement corrélée avec la surface réelle bâtie et le nombre de pièces principales.\n 
En effet, sur l'échelle de la France ce facteur n'est pas très élevé. A plus petite échelle, une corrélation plus forte apparait (cas non étudié dans ce notebook)"""

"""# Partie Modeles Machine Learning"""


"""## Apprentissage non supervisé"""


"""### Méthode du coude"""



from sklearn.cluster import KMeans
# Méthode du coude pour déterminer le nombre optimal de clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled[['Valeur fonciere', 'Surface reelle bati', 'Nombre pieces principales', 'Surface terrain', 'Date mutation']])
    sse.append(kmeans.inertia_)

fig = plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Nombre de Clusters')
plt.ylabel('SSE')
plt.title('Méthode du Coude')
plt.show()
st.pyplot(fig)

"""
On ne trouve pas de coude net sur le graphique, essayons avec la méthode de la silhouette.
"""

"""### Méthode de la silhouette"""
if (False):
    #on garde que 50% des données pour la méthode de la silhouette
    df_scaled_sample = df_scaled.sample(frac=0.5, random_state=42)

    #on ne prend pas les colonnes encodées
    df_scaled_sample = df_scaled_sample[['Valeur fonciere', 'Surface reelle bati', 'Nombre pieces principales', 'Surface terrain', 'Date mutation']]

    from sklearn.metrics import silhouette_score
    silhouette_scores = []
    with st.spinner("Cela peut prendre quelques secondes..."):
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_scaled_sample)
            silhouette_scores.append(silhouette_score(df_scaled_sample, kmeans.labels_))

    fig = plt.figure(figsize=(10, 6))
    plt.plot(range(2, 10), silhouette_scores, marker='o')
    plt.title("Silhouette Scores vs. Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    st.pyplot(fig)

#plus rapide avec image
st.image("img/silhouette.png")

"""On observe clairement un k idéal à 5 clusters"""

"""### KMeans"""

#clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42, init='k-means++')
kmeans.fit(df_scaled[['Valeur fonciere', 'Surface reelle bati', 'Nombre pieces principales', 'Surface terrain', 'Date mutation']])
labels = kmeans.labels_

#plot valeur fonciere against sufrace bati
fig = plt.figure(figsize=(10, 6))
sns.scatterplot(x='Valeur fonciere', y='Surface reelle bati', hue=labels, data=df)
plt.ylim(0, 400)
plt.xlim(0, 5000000)
plt.title('Valeur Fonciere vs Surface Réelle Bati')
plt.xlabel('Valeur Fonciere')
plt.ylabel('Surface Réelle Bati')
plt.show()
st.pyplot(fig)

#boxplot de chaque colonne avec label
fig = plt.figure(figsize=(10, 6))
sns.boxplot(x='Valeur fonciere', data=df, hue=labels)
plt.show()
st.pyplot(fig)

fig = plt.figure(figsize=(10, 6))
sns.boxplot(x='Surface reelle bati', data=df, hue=labels)
plt.show()
st.pyplot(fig)

fig = plt.figure(figsize=(10, 6))
sns.boxplot(x='Surface terrain', data=df, hue=labels)
plt.show()
st.pyplot(fig)

fig = plt.figure(figsize=(10, 6))
sns.boxplot(x='Nombre pieces principales', data=df, hue=labels)
plt.show()
st.pyplot(fig)

fig = plt.figure(figsize=(10, 6))
sns.boxplot(x='Type local', data=df, hue=labels)
plt.show()
st.pyplot(fig)

fig = plt.figure(figsize=(10, 6))
sns.boxplot(x='Date mutation', data=df, hue=labels)
plt.show()
st.pyplot(fig)


#print how many points in each cluster
for i in range(5):
    print(f"Cluster {i}: {np.sum(labels == i)} points")
    st.write(f"Cluster {i}: {np.sum(labels == i)} points")

"""On remarque plusieurs tendances intéressantes dans les clusters:\n
- cluster 0: semble s'être formé autour de la date de mutation (début d'année). Il comprend environ 40% des données du dataset\n
- cluster 2: comprend très peu de points mais uniquement des locaux commerciaux et industriels avec une très grande surface\n
- cluster 3: contient les biens à forte valeur foncière (de tous types)\n

Le score de silhouette assez faible ne permet pas une analyse plus approfondie avec kmeans."""

"""## Apprentissage supervisé"""

"""### Régression random forest"""

"""On cherche ici à prédire la valeur foncière d'un bien en fonction des autres variables du dataset. On utilisera une régression random forest pour cela.\n
On testera la fiabilité de la prédiction en entrainant le modèle sur 5% des données et en le testant sur les 95% restants, pour une question de temps et de poids du modèle"""
# Utilisation d'une régression random forest

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Sélection des features et de la cible pour la régression linéaire
X = df_encoded.drop('Valeur fonciere', axis=1)
y = df_encoded['Valeur fonciere']

#shuffle df_encoded
df_encoded = df_encoded.sample(frac=1, random_state=42)

# Division des données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
col_rf = X_train.columns

# Initialisation et entrainement du modèle de régression linéaire
from sklearn.ensemble import RandomForestRegressor

if (False):
    with st.spinner("Cela peut prendre quelques secondes..."):
        rf = RandomForestRegressor(n_estimators=100, random_state=1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

    print("Random Forest Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
    print("Random Forest R^2 Score:", r2_score(y_test, y_pred_rf))
    st.write("Random Forest Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
    st.write("Random Forest R^2 Score:", r2_score(y_test, y_pred_rf))

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.xlabel('Valeur Fonciere Réelle')
    plt.ylabel('Valeur Fonciere Prédite')
    plt.title('Random Forest : Valeur Fonciere Réelle vs Prédite')
    plt.show()
    st.pyplot(plt)

    #pickle the model
    import pickle
    filename = 'rf.pkl'
    pickle.dump(rf, open(filename, 'wb'))

st.image("img/rf.png")


"""La prédiction n'est pas très fiable, plus de données sont nécessaire pour améliorer la précision, comme la position géographique du bien, l'année de construction, etc."""

"""Vous pouvez tout de même essayer de prédire la valeur foncière d'un bien sur la page Land value - Prediction"""