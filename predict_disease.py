# Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Reading the train.csv by removing the
# last column since it's an empty column
DATA_PATH = "datasets/Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Encoding the target value into numerical
# value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

#Splitting data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24)

import pickle

if (False):
    # Training the models on whole data
    final_svm_model = SVC()
    final_nb_model = GaussianNB()
    final_rf_model = RandomForestClassifier(random_state=18)
    final_svm_model.fit(X, y)
    final_nb_model.fit(X, y)
    final_rf_model.fit(X, y)

    #pickle
    pickle.dump(final_svm_model, open('disease_svm.pkl', 'wb'))
    pickle.dump(final_nb_model, open('disease_nb.pkl', 'wb'))
    pickle.dump(final_rf_model, open('disease_rf.pkl', 'wb'))

final_svm_model = pickle.load(open('disease_svm.pkl', 'rb'))
final_nb_model = pickle.load(open('disease_nb.pkl', 'rb'))
final_rf_model = pickle.load(open('disease_rf.pkl', 'rb'))

symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms):

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1, -1)

    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    # Convert predictions to numerical representation using np.unique
    predictions, counts = np.unique([rf_prediction, nb_prediction, svm_prediction], return_counts=True)

    # Find the most frequent prediction (mode)
    final_prediction = predictions[np.argmax(counts)]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions




import streamlit as st

st.title("Disease Prediction")
st.write("This app predicts the disease based on the symptoms")
st.write("It uses three models: Random Forest, Naive Bayes, and Support Vector Machine trained on a dataset similar to this one : [Kaggle](https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning)")
st.write("Check the code behind this here: [Github](%s)" % "https://github.com/AntoineCmbld/disease_detection_ml")

with st.spinner("Predicting..."):
    selection = st.multiselect("Select the symptoms", symptom_index)

    if len(selection) == 0:
        st.write("Please select the symptoms")
    else:
        st.write("### You are likely to have: ")
        with st.container(border=True):
            st.write(predictDisease(selection)["final_prediction"])