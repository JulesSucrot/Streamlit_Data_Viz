#use pickle rf.pkl to predict land value

import pickle
import streamlit as st
import pandas as pd

#import codes and local
codes = pd.read_pickle('datasets/codes.pkl')
local = pd.read_pickle('datasets/local.pkl')

dtypes = {
    'Date mutation': 'str',
    'Valeur fonciere': 'float',
    'Code departement': 'str',
    'Type local': 'str',
    'Surface reelle bati': 'float',
    'Nombre pieces principales': 'Int64',
    'Surface terrain': 'float',
}

st.title('Land value prediction')

st.write('This app predicts the value of a land based on the following features:')
st.write('Transfer month, Department code, Type of premises, Built surface, Number of rooms, Land surface')

st.write('Please enter the following features:')

date = st.slider('Transfer month', 1, 12)
code_departement = st.selectbox('Department code', ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '21', '22', '23', '24', '25', '26', '27', '28', '29', '2A', '2B', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '971', '972', '973', '974', '976'])
#format_func to translate maison to house, appartement to apartment, local industriel. commercial ou assimilé to industrial, commercial or similar

def translate_local(option):
    return 'House' if option == 'Maison' else 'Apartment' if option == 'Appartement' else 'Industrial, commercial or similar'

type_local = st.radio('Type of premises', ['Maison', 'Appartement', 'Local industriel. commercial ou assimilé'], format_func=translate_local)
surface_reelle_bati = st.slider('Built surface', 0, 1000)
nombre_pieces_principales = st.slider('Number of rooms', 0, 10)
surface_terrain = st.slider('Land surface', 0, 1000)

oh_code = pd.DataFrame([[0]*97], columns=codes)
oh_code[f"Code departement_{code_departement}"] = 1

oh_type = pd.DataFrame([[0]*3], columns=local)
oh_type[f"Type local_{type_local}"] = 1

features = pd.DataFrame([[date, surface_reelle_bati, nombre_pieces_principales, surface_terrain]], columns=['Date mutation', 'Surface reelle bati', 'Nombre pieces principales', 'Surface terrain'])
features = pd.concat([features, oh_code], axis=1)
features = pd.concat([features, oh_type], axis=1)

model = pickle.load(open('rf.pkl', 'rb'))

prediction = model.predict(features)
#round to 2 decimal places
prediction[0] = round(prediction[0], 2)

st.write(f'### The estimated value of the land is {prediction[0]} €')

"""This value may be far from reality, as it is based on a random forest model with not many features"""