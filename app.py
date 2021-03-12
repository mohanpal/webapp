
import streamlit as st 
import pandas as pd 
import numpy as np 
import pydeck as pdk 
import altair as alt 
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import h5py
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image

st.title('Electro Base App')

image = Image.open('main-img.jpg')
st.image(image, use_column_width=True)
st.write('Please use side bar for different viz.')

df = pd.read_csv("final_data.csv") 
df_dollar = pd.read_csv("Data/AUD_USD Historical Data.csv")
df_cases = pd.read_csv("Data/covid cases.csv")
data = {'Type of Applicance':  ['Battery solar powered combined refrigerator and freezer', 
                                'waterpacks freezer',
                                'Solar direct drive combined refrigerator',
                                'Two mode battery solar refrigerator OR freezer',
                                'Vaccine/waterpacks freezer'],
        'Electricity needed(kwh/day)': ['0.99' ,'1.66','0.81','2.35','3.95']
        
        }
df_covid_elec = pd.DataFrame (data, columns = ['Type of Applicance','Electricity needed(kwh/day)'])


st.sidebar.title('Menu')

cols = st.sidebar.selectbox("Choose the Viz.", [
                                    'TOTAL DEMAND AND PREDICTION','CONSUMPTION','RRP','WEATHER','DOLLAR', 'COVID Cases'])
if cols == 'CONSUMPTION':
    st.title("Total consumption over time")
    st.sidebar.write("Consumption = Demand * Time in use. ")
    st.sidebar.write("An area with more consumption will generate more demand. more consumption indicates more activity in a area. POST-COVID it is important to understand the consumption as it can is gonna be a good indicator of economic activities")
     
if cols == 'RRP':
    st.title("Change in Regional Reference price over time")
    st.sidebar.write(" Regional Reference Price - Spot price at the regional reference node. There are 5 regions in Victoria " )
if cols == 'WEATHER':
    st.title("Change in demand with increase in temperature")
    st.sidebar.write(" Weather is one of the key drivers, with increase in temperature the demand increases and can push the price towards an upward trend" )
if cols == 'TOTAL DEMAND AND PREDICTION':
    st.title("Demand prediction using Neural network")
    st.sidebar.write(" This prediction has been made using LSTM Neural network. There are several factors affecting the demand such as consumption, temperature. As the offices were closed due to COVID-19 the demand drops of and this trend will continue with offices being closed for rest of the year. COVID-19 has decreased the demand of electricity." )
if cols == 'DOLLAR':
    st.title("Change in Demand with every 50 cent change in dollar price")
    st.sidebar.write(" Stock market is a good predictor of an countries economic condition. Every 1 percent increase in GDP growth is associated with an increase in energy consumption of 0.51 percent in the long-run. Like Electricity market , dollar price is super volatile and keep on changing every second.  " )
if cols == 'COVID Cases':
    st.title("Change in Demand with respect to COVID CASES")
    st.sidebar.write(" Each COVID case demand certain amount of electricity. There are several equipments, assays, PCR kits, machines used for a single test. Increase in COVID cases can impact the electricity demand as with eachincrase case, more hospital beds, more ICU beds, more test kits will be required. A single ventilator requires approximately 38W." )
    st.table(df_covid_elec)


fig = go.Figure()
if cols == 'CONSUMPTION':
	fig.add_trace(go.Scatter(x=df.Date, y=df.DAILYT,
                mode='lines',
                name='Total consumption'))
if cols == 'RRP':
	fig.add_trace(go.Scatter(x=df.Date, y=df.RRP,
	                    mode='markers', name='Retail rate with time'))
if cols == 'WEATHER':
	fig.add_trace(go.Scatter(x=df.TOTALDEMAND, y=df.MaxTemp,
	                    mode='markers',
	                    name='Temperature Vs Demand'))


if cols == 'TOTAL DEMAND AND PREDICTION':
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index(['Date'], inplace=True)
    scaler = MinMaxScaler()
    data_training = scaler.fit_transform(df)
    features = data_training
    target = data_training[:,0]
    TimeseriesGenerator(features, target, length=50, sampling_rate=1, batch_size=1)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, shuffle=False)
    win_length=30
    batch_size=32
    num_features = 5
    train_generator = TimeseriesGenerator(x_train,y_train,length=win_length,sampling_rate=1,batch_size=batch_size)
    test_generator = TimeseriesGenerator(x_test,y_test,length=win_length,sampling_rate=1,batch_size=batch_size)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape= (win_length,num_features), return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(64, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1))
    model = tf.keras.models.load_model('model.h5')
    model.evaluate_generator(test_generator, verbose=0)
    predictions = model.predict_generator(test_generator)
    df_pred = pd.concat([pd.DataFrame(predictions), pd.DataFrame(x_test[:,1:][win_length:])],axis=1)
    rev_trans = scaler.inverse_transform(df_pred)
    df_new=df[predictions.shape[0]*-1:]
    df_new['Energy_pred'] = rev_trans[:,0]
    df_new['TOTALDEMAND'] = df_new.round(2)
    fig = px.line(df_new, x=df_new.index, y=['TOTALDEMAND','Energy_pred'])

    #fig.add_scatter(x=df_new.index, y=df_new['Energy_pred'], mode='lines')
    # for trace in fig.data:
    #     trace.name = NA
    #df_new[['TOTALDEMAND','Energy_pred']].plot()

   # fig.add_trace(go.Scatter(x=df_new.Date, y=df_new.Energy_pred,
	#                mode='lines+markers',
	 #               name='Recoveries'))
if cols == 'DOLLAR':
	fig.add_trace(go.Scatter(x=df.TOTALDEMAND, y=df_dollar.Price,
	                    mode='markers',
	                    name='Dollar Vs Demand'))
if cols == 'COVID Cases':
	fig.add_trace(go.Scatter(x=df.TOTALDEMAND, y=df_cases.Cases,
	                    mode='markers'))
st.plotly_chart(fig, use_container_width=True)




  

