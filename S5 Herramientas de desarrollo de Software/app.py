import streamlit as st
import pandas as pd
import plotly.express as px

car_data = pd.read_csv('vehicles_us.csv')

st.title('Analysing Car sales dataset')

st.header('Mark the checkbox to build Histogram or Scatterplot')

build_histogram = st.checkbox('Build Histogram')
build_scatter = st.checkbox('Build Scatterplot')

if build_histogram:
    st.write('Build histogram for vehicle sales dataset')
    fig = px.histogram(car_data, x='odometer')
    st.plotly_chart(fig)

if build_scatter:
    st.write('Building scatterplot for vehicle sales dataset')
    fig2 = px.scatter(car_data, x='odometer', y='price')
    st.plotly_chart(fig2)


st.header('Referenced dataframe  ')
st.dataframe(car_data.head(5))

st.header('Github repository link')
st.link_button('Github repository', "https://github.com/OmiHdz/project_s5.git")


st.header('Readme')
st.link_button(
    'Go', "https://github.com/OmiHdz/project_s5/blob/main/README.md")

st.caption('This script was coded by Omar Hernández® for tripleten® Data Science bootcamp ,all rights reserved. ')
#
