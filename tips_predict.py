import pickle
import streamlit as st
import pandas as pd

# import codes and local
model = pickle.load(open('tip_predict.pkl', 'rb'))

st.title('Tips prediction')

st.write('This app predicts the tip based on the following features:')
st.write('Total bill, day of the week, time of the day, number of people at the table')

st.write('Please enter the following features:')
total_bill = st.slider('Total bill', 0, 100)
day = st.selectbox('Day of the week', range(4), format_func=lambda x: ['Thur', 'Fri', 'Sat', 'Sun'][x])
time = st.radio('Time of the day', [0, 1], format_func=lambda x: 'Dinner' if x == 0 else 'Lunch')
size = st.slider('Number of people at the table', 1, 10)

features = pd.DataFrame([[total_bill, day, time, size]], columns=['total_bill', 'day', 'time', 'size'])

prediction = model.predict(features)
# round to 2 decimal places
prediction[0] = round(prediction[0], 2)

st.write(f'### The estimated tip is {prediction[0]} €')

