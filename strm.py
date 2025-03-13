import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Tip Prediction')
st.write('This app predicts the tip amount based on the bill amount and the quality of service')

total_bill=st.number_input('Toatl Bill')
smoker=st.radio('Smoker',('Yes','No'))
day=st.radio('Day',('Sun', 'Sat', 'Thur', 'Fri'))
size=st.number_input('Number of customers')
sex=st.selectbox('Gender',['Male','Female'])
time=st.selectbox('Time',['Dinner','Lunch'])

input_data={
    'total_bill': total_bill,
    'sex': sex,
    'smoker': smoker,
    'day': day,
    'size': size,
    'time': time
}

model=pickle.load(open('model.pkl','rb'))

new_data=pd.DataFrame([input_data])
 
new_data['smoker']=[0 if smoker=='No' else 1 ]

en=pickle.load(open('Lenco.pkl','rb'))
new_data['day']=en.transform(new_data['day'])

s=pd.get_dummies(new_data['sex'],dtype=int)
new_data=pd.concat([new_data,s],axis=1)
new_data=new_data.drop('sex',axis=1)

t=pd.get_dummies(new_data['time'],dtype=int)
new_data=pd.concat([new_data,t],axis=1)
new_data=new_data.drop('time',axis=1)

df=pd.read_csv('features.csv')
columns_list = [col for col in df.columns if col != 'Unnamed: 0']
new_data=new_data.reindex(columns=columns_list,fill_value=0)

prediction=model.predict(new_data)[0]
if st.button ('Predict'):
     st.write('Predicted Tip:',prediction)
