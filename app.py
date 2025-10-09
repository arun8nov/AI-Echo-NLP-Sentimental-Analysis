# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
import pickle
import streamlit as st
from dash import preprocss

PR= preprocss()


# Setting Page Configuration
st.set_page_config(page_title='AI Echo: Your Smartest Conversational Partner', 
                   layout='wide',page_icon='AI')

def Heder():
    c1,c2 = st.columns([0.3,2])
    c1.image('logo.png')
    c2.title("AI Echo Sentimental Analysis") # Seeting up Logo and Title

# Read dataset from google sheet
sheet_id = "1eyPDJj8ttd8t-o6JVT4txCbvJ9DtcF-U"
sheet_name = "Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

data = pd.read_csv(url)


def Data():
    Heder()
    df = PR.data_clean(data)
    st.dataframe(df)
    st.divider()
    h1,h2 = st.columns([1.3,2])
    h1.text(' ')
    h2.header("Table Data Analytics")

    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(PR.rat_dis(df))
    c2.plotly_chart(PR.rat_ve_pu(df))
    c3.plotly_chart(PR.vote_rat(df))

    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(PR.lan_count(df))
    c2.plotly_chart(PR.lan_rat(df))
    c3.plotly_chart(PR.plat_rat(df))

    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(PR.loc_rat(df))
    c2.plotly_chart(PR.loc_rat_VP(df))
    c3.plotly_chart(PR.ver_rat(df))

    st.divider()
    c1,c2 = st.columns([1.3,2])
    c2.header("Word Data Analytics")

    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(PR.char_dis(df))
    c2.plotly_chart(PR.word_dis(df))
    c3.plotly_chart(PR.sent_dis(df))

    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(PR.char_box(df))
    c2.plotly_chart(PR.word_box(df))
    c3.plotly_chart(PR.sent_box(df))

    h1,h2 = st.columns([1.0,2])
    h1.text(' ')
    h2.subheader("Cleaned Text Content & Analysis")

    s_df = PR.text_clean(df)
    st.dataframe(s_df)

    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(PR.all_words(s_df))
    c2.plotly_chart(PR.rat_1_words(s_df))
    c3.plotly_chart(PR.rat_2_words(s_df))

    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(PR.rat_3_words(s_df))
    c2.plotly_chart(PR.rat_4_words(s_df))
    c3.plotly_chart(PR.rat_5_words(s_df))

    c1,c2 = st.columns(2)
    c1.plotly_chart(PR.frq_count(s_df))




    
    


    






pg = st.navigation([Data])
pg.run()
