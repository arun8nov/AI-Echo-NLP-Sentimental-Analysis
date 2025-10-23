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
from dash1 import chatgpt_review
# Loading the Model
model = pickle.load(open('model.pkl','rb'))
# Creating Object for Classes
PR= preprocss()
CR = chatgpt_review()


# Setting Page Configuration
st.set_page_config(page_title='AI Echo: Your Smartest Conversational Partner', 
                   layout='wide',page_icon='AI')
# Creating Header Function
def Heder():
    c1,c2 = st.columns([0.3,2])
    c1.image('logo.png') # Seeting up Logo
    c2.title("AI Echo Sentimental Analysis") # Seeting up Logo and Title

# Read dataset from google sheet
sheet_id = "1eyPDJj8ttd8t-o6JVT4txCbvJ9DtcF-U"
sheet_name = "Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

data = pd.read_csv(url)


def Main_Dash():
    Heder()
    df = PR.data_clean(data) # Reading Data
    st.dataframe(df) # Displaying Data
    st.divider()
    h1,h2 = st.columns([1.3,2])
    h1.text(' ')
    h2.header("Table Data Analytics")
    # Plotting Various Graphs
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

    st.plotly_chart(PR.frq_count(s_df))
# Defining kaggle downloded file  Dashboard
def Gpt_Dash():
    Heder()
    # Reading Data
    df = pd.read_csv("D:\GIT\AI-Echo-NLP-Sentimental-Analysis\data\chatgpt_reviews_en_clean.csv",index_col=0)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    st.dataframe(df)
    st.divider()
    # Plotting Various Graphs
    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(CR.lab_dis(df))
    c2.plotly_chart(CR.rat_dis(df))
    c3.plotly_chart(CR.rat_pie(df))
    
    c1,c2 = st.columns(2)
    c1.plotly_chart(CR.char_dis(df))
    c2.plotly_chart(CR.word_dis(df))

    c1,c2 = st.columns(2)
    c1.plotly_chart(CR.char_box(df))
    c2.plotly_chart(CR.word_box(df))

    c1,c2 = st.columns(2)
    c1.plotly_chart(CR.char_average(df))
    c2.plotly_chart(CR.word_average(df))

    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(CR.rat_comp(df))
    c2.plotly_chart(CR.top_users(df))
    c3.plotly_chart(CR.thump_up(df))

    c1,c2 = st.columns([0.5,2])
    c1.plotly_chart(CR.review_over_year(df))
    c2.plotly_chart(CR.review_over_month(df))

    st.plotly_chart(CR.time_Analysis(df))

    f1,f2,f3 = CR.overall_wordcloud(df)
    c1,c2,c3 = st.columns(3)
    c1.plotly_chart(f1)
    c2.plotly_chart(f2)
    c3.plotly_chart(f3)

    st.plotly_chart(CR.frq_count_over_rating(df))
    st.plotly_chart(CR.frq_count_over_label(df))

# Defining Review Classification Dashboard
def Classification():
    
    Heder()


    # Creating Form for User Input
    with st.form(key= 'Review Classification'):

        text_data = st.text_input('Enter Review') # Input Text for Classification

        submit_button = st.form_submit_button(label='submit') # Submit Button

        if submit_button:
            result = model.predict(pd.Series(text_data))[0] # Predicting the Result
            # Displaying the Result
            if result == 'pos':
                st.success('Positive')
            elif result == 'neu':
                st.success('Neutral')
            else:
                st.error('Negative')

    
# Creating Page Navigation
pg = st.navigation([Main_Dash,Gpt_Dash,Classification],position='top')
pg.run()
