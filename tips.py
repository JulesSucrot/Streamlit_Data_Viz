#import tips.csv
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Tips analysis')

st.write('This app analyzes a database of tips from a restaurant. The database contains the following columns:')

#read pickle
df = pd.read_pickle('datasets/tips.pkl')
dtypes = df.dtypes
col_names = df.columns
#dont show row index
df_show = pd.DataFrame({'Column name': col_names, 'Data type': dtypes}).reset_index(drop=True)
st.write(df_show)

if (False):
    df = pd.read_csv('datasets/tips.csv')
    #pickle the dataframe
    df.to_pickle('datasets/tips.pkl')

st.write("## Basic statistics")
# Mean, median, and distribution of total_bill and tip.
st.write("### Total bill")
st.write(f"Mean: {round(df['total_bill'].mean(), 2)}€")
st.write(f"Median: {round(df['total_bill'].median(), 2)}€")
st.write(f"Distribution:")
fig = plt.figure()
sns.histplot(df['total_bill'], kde=True)
st.pyplot(fig)

#plot tip against total_bill
st.write("### Tip against total bill")
fig = plt.figure()
sns.scatterplot(data=df, x='total_bill', y='tip')
st.pyplot(fig)

#Tip percentage (tip / total_bill).
df['tip_percentage'] = df['tip'] / df['total_bill']
st.write("### Tip percentage (tip / total_bill)")
#round to 2 secimals
st.write(f"Mean: {round(df['tip_percentage'].mean()*100, 2)}%")
st.write(f"Median: {round(df['tip_percentage'].median()*100, 2)}%")

#encoding
import sklearn.preprocessing as skp
label_encoder = skp.LabelEncoder()
df_encoded = df.copy()
df_encoded.drop(columns=['tip_percentage'], inplace=True)
df_encoded["sex"] = label_encoder.fit_transform(df_encoded["sex"])
df_encoded["smoker"] = label_encoder.fit_transform(df_encoded["smoker"])
df_encoded["time"] = label_encoder.fit_transform(df_encoded["time"])
day_encoder = {"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3}
df_encoded["day"] = df_encoded["day"].map(day_encoder)

#standardize
scaler = skp.StandardScaler()
df_standard = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

st.write("## Correlation Analysis")
st.write("### Correlation matrix")
#correlation matrix
fig = plt.figure()
sns.heatmap(df_standard.corr(), annot=True, cmap='coolwarm')
st.pyplot(fig)

"""As expected, the tip is highly correlated with the total bill and the size of the table. The day of the week, time of the day and sex are a little correlated with the tip or the total bill. The smoker status is not correlated with the tip or the total bill."""

st.write("### Aggregated statistics")
#Average tip and total_bill grouped by day, time, sex, smoker and table size
st.write("#### Average tip and total bill grouped by day")
grouped = df.groupby('day')[['tip', 'total_bill']].mean()
#plot tip and total_bill grouped by day on same barplot
fig = plt.figure()
#show in the order of the days of the week
grouped = grouped.reindex(['Thur', 'Fri', 'Sat', 'Sun'])
grouped.plot(kind='bar', ax=plt.gca())
st.pyplot(fig)

st.write(grouped)


"""We can clearly see that the average tip is higher on Sunday, followed by Saturday. The average total bill follows this trend as well."""

st.write("#### Average tip and total bill grouped by time")
grouped = df.groupby('time')[['tip', 'total_bill']].mean()
#plot tip and total_bill grouped by time on same barplot
fig = plt.figure()
grouped.plot(kind='bar', ax=plt.gca())
st.pyplot(fig)
st.write(grouped)

#calculate the average tip and total bill for lunch and for dinner
lunch = df[df['time'] == 'Lunch']
dinner = df[df['time'] == 'Dinner']
lunch_tip = lunch['tip'].mean()
lunch_total_bill = lunch['total_bill'].mean()
dinner_tip = dinner['tip'].mean()
dinner_total_bill = dinner['total_bill'].mean()


f"""The average tip are {round((dinner_tip/lunch_tip - 1)*100, 2)}% and total bill are {round((dinner_total_bill/lunch_total_bill - 1)*100, 2)}% higher during dinner time compared to lunch time."""

st.write("#### Average tip and total bill grouped by sex")
grouped = df.groupby("sex")[["tip", "total_bill"]].mean()
#plot tip and total_bill grouped by sex on same barplot
fig = plt.figure()
grouped.plot(kind='bar', ax=plt.gca())
st.pyplot(fig)
st.write(grouped)

#calculate the average tip and total bill for men and women
man = df[df['sex'] == 'Male']
woman = df[df['sex'] == 'Female']
man_tip = man['tip'].mean()
man_total_bill = man['total_bill'].mean()
woman_tip = woman['tip'].mean()
woman_total_bill = woman['total_bill'].mean()


f"""Male clients tend to leave a {round((man_tip/woman_tip-1)*100, 2)}% higher tip and have a {round((man_total_bill/woman_total_bill - 1)*100, 2)}% higher total bill than female clients."""

st.write("#### Average tip and total bill grouped by smoker")
grouped = df.groupby("smoker")[["tip", "total_bill"]].mean()
#plot tip and total_bill grouped by smoker on same barplot
fig = plt.figure()
grouped.plot(kind='bar', ax=plt.gca())
st.pyplot(fig)
st.write(grouped)

"""There is no significant difference in the average tip or total bill between smokers and non-smokers."""

st.write("#### Average tip and total bill grouped by table size")
grouped = df.groupby("size")[["tip", "total_bill"]].mean()
#plot tip and total_bill grouped by table size on same barplot
fig = plt.figure()
grouped.plot(kind='bar', ax=plt.gca())
st.pyplot(fig)

#calculate tip per person
grouped['tip_per_person'] = grouped['tip'] / grouped.index
grouped['bill_per_person'] = grouped['total_bill'] / grouped.index
st.write(grouped)

"""Very interesting, the less people there are at the table, the more they tend to leave a tip. However, the total bill per person reaches its peak for tables of 2 people."""

st.write("## Conclusion")
st.write("This analysis of the tips database shows that the total bill and tip are highly correlated. The day of the week, time of the day, sex and table size all have a small impact on the average tip per person. The ideal table size for the restaurant is 2 people, as it generates the highest total bill per person. People that eat alone tend to leave a higher tip.")

st.write("# Machine learning model to predict the tip")

#use linear regression to predict the tip

import sklearn.model_selection as skm
import sklearn
from sklearn.linear_model import LinearRegression

X = df_encoded.drop(columns=['tip', 'smoker', 'sex'])
y = df_encoded['tip']
X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

#evaluate the model
y_pred = model.predict(X_test)
mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
r2 = sklearn.metrics.r2_score(y_test, y_pred)

st.write(f"The mean squared error is {round(mse, 2)}")
st.write(f"The mean absolute error is {round(mae, 2)}")
st.write(f"The R² score is {round(r2, 2)}")

#plot prediction against true value with x=y line
fig = plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([0, 10], [0, 10], color='red')
plt.xlabel('True value')
plt.ylabel('Predicted value')
st.pyplot(fig)

#pickling the model
if False:
    import pickle
    pickle.dump(model, open('tip_predict.pkl', 'wb'))
