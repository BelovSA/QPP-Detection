import os
import numpy as np
import pandas as pd
import streamlit as st
from Model import make_predictions 

st.set_page_config(
        page_title="QPP detection",
)

st.title("QPP detection")

st.text_input("Type the path to your folder with .csv (or simply drag the folder)", key="path")

if st.button('Detect'):
    path = st.session_state.path

    preds = make_predictions(path)
    st.dataframe(
        preds,
        column_config={
        "Files": "Files",
        "proba": st.column_config.NumberColumn( "QPP confidence"),
        "QPP": "QPP",
        'Figs': st.column_config.LineChartColumn( "Flare profiles", width="large",)
},
        hide_index=True,
        use_container_width=False
)
    
    csv = preds.drop(columns=['Figs']).to_csv()
 
    st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='predictions.csv',
    mime='text/csv',
)

        

