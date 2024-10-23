import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd

# Streamlit app setup
st.title('Blood Donation Data Analysis')

# Link to the dataset
st.markdown('[Blood Donation Data - Locations and Times](https://www.data.gouv.fr/fr/datasets/don-du-sang-lieux-et-horaires-des-collectes/)')

# Load the dataset to explore its content
file_path = 'datasets/collections.csv'
data = pd.read_csv(file_path)

# Display the first few rows and general info about the dataset
st.header('Dataset Overview')
st.write(data.head())

# Dropping columns with too many missing values that are not useful for the analysis
columns_to_drop = [
    'location_phone', 'location_postCode', 'location_horaires', 
    'location_infos', 'location_metro', 'location_bus', 'location_tram', 
    'location_parking', 'location_debutInfos', 'location_finInfos'
]
data_cleaned = data.drop(columns=columns_to_drop)

# Checking for any further null values and converting the 'collection_date' to datetime format for time-based analysis
data_cleaned['collection_date'] = pd.to_datetime(data_cleaned['collection_date'], errors='coerce')

# Plotting the count of different types of donations
donation_types = ['location_giveBlood', 'location_givePlasma', 'location_givePlatelet']
donation_counts = data_cleaned[donation_types].sum()

st.header('Count of Different Donation Types')
fig, ax = plt.subplots(figsize=(8, 5))
donation_counts.plot(kind='bar', ax=ax)
ax.set_title('Count of Different Donation Types')
ax.set_ylabel('Number of Locations Offering Donation Type')
ax.set_xticklabels(donation_counts.index, rotation=0)
st.pyplot(fig)

# Group by public and non-public collections and count the occurrences
public_vs_non_public = data_cleaned['collection_isPublic'].value_counts()

# Streamlit section
st.title("Public vs Non-Public Blood Donation Collection Events")

# Plotting the data
fig, ax = plt.subplots()
public_vs_non_public.plot(kind='bar', ax=ax)
ax.set_title('Comparison of Public vs. Non-Public Collection Events')
ax.set_xlabel('Is Public Event')
ax.set_ylabel('Number of Collection Events')
ax.set_xticklabels(['Public', 'Non-Public'], rotation=0)

# Display the chart on Streamlit
st.pyplot(fig)


# Création d'une copie des données nettoyées pour éviter de modifier l'original
data_cleaned = data_cleaned.copy()

# Créer un DataFrame pour contenir les informations de longitude et latitude
df_locations = data_cleaned[['location_longitude', 'location_latitude']].dropna()

# Création de la carte en utilisant Plotly Express
st.header('Geographical Distribution of Donation Locations')
fig = px.scatter_mapbox(
    df_locations,
    lat='location_latitude',
    lon='location_longitude',
    zoom=4,
    center={"lat": 46.603354, "lon": 1.888334},
    mapbox_style='carto-positron',
    opacity=0.5,
    title='Geographical Distribution of Donation Locations'
)

# Afficher la carte sur Streamlit
st.plotly_chart(fig)

# Time-based analysis of blood donation collections
st.header('Number of Blood Donation Collections Over Time')
fig, ax = plt.subplots(figsize=(10, 6))
data_cleaned['collection_date'].dt.to_period('M').value_counts().sort_index().plot(kind='line', ax=ax)
ax.set_title('Number of Blood Donation Collections Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Number of Collections')
ax.grid(True)
st.pyplot(fig)

# Selecting relevant categorical features for MCA analysis
categorical_features = ['location_regionName', 'location_giveBlood', 'location_givePlasma', 'location_givePlatelet']

# Plotting the distribution of donation times
time_columns = ['collection_morningStartTime', 'collection_morningEndTime', 'collection_afternoonStartTime', 'collection_afternoonEndTime']

# Dropping rows with NaT values in time columns for valid analysis
data_cleaned[time_columns] = data_cleaned[time_columns].apply(pd.to_datetime, errors='coerce')

# Filtering only valid rows for time analysis
valid_time_data = data_cleaned.dropna(subset=time_columns)

# Morning donation times
morning_start = valid_time_data['collection_morningStartTime']
morning_end = valid_time_data['collection_morningEndTime']

st.header('Distribution of Morning and Afternoon Donation Times')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.hist([morning_start.dt.hour, morning_end.dt.hour], bins=range(5, 13), label=['Morning Start', 'Morning End'], alpha=0.7)
ax1.set_title('Distribution of Morning Donation Times')
ax1.set_xlabel('Hour of the Day')
ax1.set_ylabel('Frequency')
ax1.legend()

# Afternoon donation times
afternoon_start = valid_time_data['collection_afternoonStartTime']
afternoon_end = valid_time_data['collection_afternoonEndTime']

ax2.hist([afternoon_start.dt.hour, afternoon_end.dt.hour], bins=range(12, 21), label=['Afternoon Start', 'Afternoon End'], alpha=0.7)
ax2.set_title('Distribution of Afternoon Donation Times')
ax2.set_xlabel('Hour of the Day')
ax2.set_ylabel('Frequency')
ax2.legend()

plt.tight_layout()
st.pyplot(fig)

# Recomputing the region-wise average morning and afternoon start times, ensuring only valid time data is processed
region_donation_times = valid_time_data.groupby('location_regionName')[time_columns].apply(
    lambda x: x.apply(lambda time_col: time_col.dt.hour.mean())
)

# Calculating the mean start time by region (combining morning and afternoon start times)
region_mean_start_times = region_donation_times[['collection_morningStartTime', 'collection_afternoonStartTime']].mean(axis=1)

# Trier les temps de début moyen par région par ordre croissant
region_mean_start_times_sorted = region_mean_start_times.sort_values()

# Créer l'histogramme
st.header('Mean starting time of Donations by Region')
fig, ax = plt.subplots(figsize=(10, 6))
region_mean_start_times_sorted.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Mean Start Time of Donations by Region')
ax.set_xlabel('Region Name')
ax.set_ylabel('Mean Start Time (Hour)')
ax.set_xticklabels(region_mean_start_times_sorted.index, rotation=90)
ax.grid(axis='y')


# Afficher l'histogramme sur Streamlit
st.pyplot(fig)

region_donation_counts = data_cleaned['location_regionName'].value_counts().sort_values()

# Créer l'histogramme des comptes de donations par région
st.header('Number of Donations by Region')
fig, ax = plt.subplots(figsize=(10, 6))
region_donation_counts.plot(kind='bar', color='lightgreen', ax=ax)
ax.set_title('Number of Donations by Region')
ax.set_xlabel('Region Name')
ax.set_ylabel('Number of Donations')
ax.set_xticklabels(region_donation_counts.index, rotation=90)
ax.grid(axis='y')

# Afficher l'histogramme sur Streamlit
st.pyplot(fig)


# Filter the dataset to focus on blood donation locations and group by region name
blood_donations_by_region = data_cleaned[data_cleaned['location_giveBlood'] == True].groupby('location_regionName').size()

# Define the population data per region
population_by_region = {
    'Auvergne-Rhône-Alpes': 8235923,
    'Bourgogne- Franche-Comté': 2791719,
    'Bretagne': 3453023,
    'Centre - Val de Loire': 2573295,
    'Corse': 355528,
    'Grand Est': 5568711,
    'Hauts-de-France': 5983823,
    'Île-de-France': 12419961,
    'Normandie': 3327077,
    'Nouvelle Aquitaine': 6154772,
    'Occitanie': 6154729,
    'Pays de la Loire': 3926389,
    "Provence-Alpes-Côte d'Azur": 5198011
}

# Create a mapping to match dataset regions to population regions
region_mapping = {
    'EFS Auvergne-Rhône Alpes': 'Auvergne-Rhône-Alpes',
    'EFS Bourgogne Franche Comté': 'Bourgogne- Franche-Comté',
    'EFS Bretagne': 'Bretagne',
    'EFS Centre-Pays de la Loire': 'Centre - Val de Loire',
    'EFS Grand Est': 'Grand Est',
    'EFS Haut de France-Normandie': 'Hauts-de-France',  # Includes Normandie
    'EFS Ile de France': 'Île-de-France',
    'EFS Nouvelle Aquitaine': 'Nouvelle Aquitaine',
    'EFS Occitanie': 'Occitanie',
    'EFS PACA-Corse': "Provence-Alpes-Côte d'Azur",  # Includes Corse
}

# Map dataset regions to population regions
blood_donations_by_region_mapped = blood_donations_by_region.rename(index=region_mapping)

# Calculate the percentage of blood donation sites per population by region
blood_donations_percentage = blood_donations_by_region_mapped / pd.Series(population_by_region) * 100

# Drop regions that were not mapped
blood_donations_percentage = blood_donations_percentage.dropna()

# Streamlit app
st.title('Percentage of Population Giving Blood by Region in France')

# Plot the percentage of blood donation sites per population by region as a pie chart
fig, ax = plt.subplots(figsize=(10, 6))
blood_donations_percentage.plot(kind='pie', labels=blood_donations_percentage.index, autopct=lambda p: f'{p:.1f}%\n({p * blood_donations_percentage.sum() / 100:.2f}%)', startangle=90, ax=ax, colors=plt.cm.Paired.colors)
ax.set_ylabel('')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

st.pyplot(fig)
