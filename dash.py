import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nbformat
import re
import nltk
from nltk.tokenize import sent_tokenize
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

lemmatizer =  WordNetLemmatizer()
stopwords = stopwords.words('english')

my_col = ['#1f77b4', '#2ca02c', '#8c564b', '#7f7f7f', '#17becf']

class preprocss:
    def __init__(self):
        pass

    def data_clean(self,df):
        df = df
        df['full_review'] = df['review']
        df.drop('review_length',axis=1,inplace=True)
        df['char_len'] = df['full_review'].apply(lambda x: len(x))
        df['word_len'] = df['full_review'].apply(lambda x : len(x.split(" ")))
        df['sent_len'] = df['full_review'].apply(lambda x : len(sent_tokenize(x)))  
        return df
    
    def text_clean(self,df):
        df = df[['full_review','rating']]

        # Inner Fuction 1 for text cleaning
        def text_cleaning(text):
            # Lower Casing
            text = text.lower()
            # url Removal
            url_pat = re.compile(r'https?://\S+|www\.\S+')
            text = url_pat.sub(r'', text)
            # mail removal
            mail_pat = re.compile(r'\S+@\S+')
            text = mail_pat.sub(r'', text)
            # Remove Punctuations
            text = re.sub(r'[^\w\s]', '', text)
            # Remove Numbers
            text = re.sub(r'[A-Za-z]+\d+','',text)
            text = re.sub(r'\d+\s*[A-Za-z]+','',text)
            text = re.sub(r'\d+','',text)
            # Remove White sapace
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        
        
        # Inner Function 2 for Lemmatation
        def lemma(text):
            text = word_tokenize(text)
            text =  [contractions.fix(word) for word in text]
            text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords]
            text = ' '.join(text)
            return text
        df['final_text'] = df['full_review'].apply(text_cleaning).apply(lemma)

        df = df[['final_text','rating']]

        return df
    

    def rat_dis(self,df):
        fig = px.histogram(df,
                   x='rating',
                   title='Distribution of Rating',
                   color='rating',
                   color_discrete_sequence=my_col
                   )
        fig.update_layout(
            title ={'x':0.25}
        )

        return fig

    def vote_rat(self,df):

        t_df = df.groupby('rating')['helpful_votes'].mean().reset_index()
        fig = px.pie(t_df,
            values='helpful_votes',
            names='rating',
            hole=0.5,
            hover_name='rating',
            title= 'Average Helpfull voting of ratings',
            color_discrete_sequence=my_col
            )
        fig.update_layout(
            title ={'x':0.25},
            showlegend = False
        )

        return fig
    
    def loc_rat(self, df):

        t_df = df.groupby('location')['rating'].mean().reset_index().sort_values(by='location',ascending=False)
        fig = px.bar(t_df,
            x='rating',
            y='location',
            color='rating',
            title = 'Average Rating by user location',
            color_continuous_scale=my_col
        )
        fig.update_layout(
            title ={'x':0.25},
            height = 800
        )

        return fig
    
    def loc_rat_VP(self,df):
        t_df = df[['location','verified_purchase','rating']]
        t_df_2 = t_df[t_df['verified_purchase'] == 'Yes']
        t_df_1 = t_df[t_df['verified_purchase'] == 'No']
        t_df_1['rating'] = t_df_1['rating'] * -1
        t_df = pd.concat([t_df_1,t_df_2],axis=0)

        fig = px.bar(t_df,
                x = 'rating',
                y = 'location',
                color= 'verified_purchase',
                color_discrete_map = {'Yes': '#2ca02c', 'No': '#8c564b'},
                title='Location Based Raitng by Verified Purchase'
                )

        fig.update_layout(
            height = 800,
            showlegend = False,
            title = {'x':0.25}
                )
        
        return fig
    
    def lan_count(self,df):

        t_df = df['language'].value_counts().reset_index()
        fig =  px.bar(t_df,
                    x = 'language',
                    y = 'count',
                    color= 'count',
                    title='Language base reiewing count',
                    color_continuous_scale=my_col
                    )
        fig.update_layout(
            title ={'x':0.25},
        )
        
        return fig
    
    def lan_rat(sel,df):

        t_df = df.groupby('language')['rating'].mean().reset_index().sort_values(by='rating',ascending=False)
        fig = px.bar(t_df,
                    y='rating',
                    x='language',
                    color='rating',
                    color_continuous_scale=my_col,
                    title='Average Rating by user language')
        fig.update_layout(
            title ={'x':0.25},
        )

        return fig
    
    def plat_rat(self,df):

        t_df = df.groupby('platform')['rating'].mean().reset_index().sort_values(by='rating',ascending=False)

        fig = px.bar(t_df,
                    x='platform',
                    y='rating',
                    color='rating',
                    title='Average Rating by user platform',
                    color_continuous_scale=my_col)
        fig.update_layout(
            title ={'x':0.25},
        )
        
        return fig
    
    def rat_ve_pu(self,df):
        fig = px.histogram(
                df,
                x='rating',
                color='verified_purchase',
                title='Rating Distribution based on Verified purchase',
                color_discrete_sequence=my_col
            )
        fig.update_layout(
            title ={'x':0.25},
        )

        return fig
    
    def ver_rat(self,df):
        t_df = df.groupby('version')['rating'].mean().reset_index().sort_values(by=['rating','version'],ascending=False)
        fig = px.funnel(t_df,
                y='version',
                x='rating',
                color='rating',
                title='versions having high rating',
                )
        fig.update_layout(
                title ={'x':0.25},
                showlegend = False,
                height = 800
            )

        return fig 
    
    def char_dis(self,df):

        fig = px.histogram(df,
                x = 'char_len',
                color='rating',
                color_discrete_sequence=my_col,
                title='Review Charecters Distribution')  
        fig.update_layout(
                title ={'x':0.25})      
       
        return fig

    def word_dis(self,df):

        fig = px.histogram(df,
                x = 'word_len',
                color='rating',
                color_discrete_sequence=my_col,
                title='Review Word Distribution')
        fig.update_layout(
                title ={'x':0.25})  
        
        return fig 
    
    def sent_dis(self,df):
        fig = px.histogram(df,
                x = 'sent_len',
                color='rating',
                color_discrete_sequence=my_col,
                title='Review Sentence Distribution')
        fig.update_layout(
                title ={'x':0.25})  
        
        return fig
    
    def char_box(self,df):
        fig = px.box(df['char_len'],
                     title='Charter',
                     orientation='h')
        fig.update_layout(
                title ={'x':0.5})
        
        return fig
    
    def word_box(self,df):
        fig = px.box(df['word_len'],
                     title='Word',
                     orientation='h')
        fig.update_layout(
                title ={'x':0.5})
        
        return fig
    
    def sent_box(self,df):
        fig = px.box(df['sent_len'],
                     title='Sentence',
                     orientation='h')
        fig.update_layout(
                title ={'x':0.5})
        
        return fig
    
    def all_words(self,df):
        all_words = ' '.join([text for text in df['final_text']])
        wordcloud=WordCloud(width=800,height=500,random_state=42,max_font_size=110).generate(all_words)
        fig = px.imshow(wordcloud,title="All word in Final Text")

        fig.update_layout(
            title ={'x':0.3}
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
       
        return fig
    
    def rat_1_words(self,df):
        rat_1_words = ' '.join([text for text in df[df['rating'] == 1]['final_text']])
        wordcloud=WordCloud(width=800,height=500,random_state=42,max_font_size=110).generate(rat_1_words)
        fig = px.imshow(wordcloud,title="Rating 1 Words")

        fig.update_layout(
            title ={'x':0.3}
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
       
        return fig
    
    def rat_2_words(self,df):
        rat_2_words = ' '.join([text for text in df[df['rating'] == 2]['final_text']])
        wordcloud=WordCloud(width=800,height=500,random_state=42,max_font_size=110).generate(rat_2_words)
        fig = px.imshow(wordcloud,title="Rating 2 Words")

        fig.update_layout(
            title ={'x':0.3}
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
       
        return fig

    def rat_3_words(self,df):
        rat_3_words = ' '.join([text for text in df[df['rating'] == 3]['final_text']])
        wordcloud=WordCloud(width=800,height=500,random_state=42,max_font_size=110).generate(rat_3_words)
        fig = px.imshow(wordcloud,title="Rating 3 Words")

        fig.update_layout(
            title ={'x':0.3}
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
       
        return fig
    
    def rat_4_words(self,df):
        rat_4_words = ' '.join([text for text in df[df['rating'] == 4]['final_text']])
        wordcloud=WordCloud(width=800,height=500,random_state=42,max_font_size=110).generate(rat_4_words)
        fig = px.imshow(wordcloud,title="Rating 4 Words")

        fig.update_layout(
            title ={'x':0.3}
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
       
        return fig
    
    def rat_5_words(self,df):
        rat_5_words = ' '.join([text for text in df[df['rating'] == 5]['final_text']])
        wordcloud=WordCloud(width=800,height=500,random_state=42,max_font_size=110).generate(rat_5_words)
        fig = px.imshow(wordcloud,title="Rating 5 Words")

        fig.update_layout(
            title ={'x':0.3}
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
       
        return fig
    
    def frq_count(self,df):
        t_df1 = pd.Series(' '.join(df[df['rating'] == 1]['final_text']).split()).value_counts().head(10).reset_index()
        t_df1['rating'] = 1
        t_df2 = pd.Series(' '.join(df[df['rating'] == 2]['final_text']).split()).value_counts().head(10).reset_index()
        t_df2['rating'] = 2
        t_df3 = pd.Series(' '.join(df[df['rating'] == 3]['final_text']).split()).value_counts().head(10).reset_index()
        t_df3['rating'] = 3
        t_df4 = pd.Series(' '.join(df[df['rating'] == 4]['final_text']).split()).value_counts().head(10).reset_index()
        t_df4['rating'] = 4
        t_df5 = pd.Series(' '.join(df[df['rating'] == 5]['final_text']).split()).value_counts().head(10).reset_index()
        t_df5['rating'] = 5

        t_df = pd.concat([t_df1,t_df2,t_df3,t_df4,t_df5],axis=0)

        fig = px.funnel(t_df,
                        y = 'count',
                        x = 'index',
                        color='rating',
                        title = 'Top 10 Repeting words in each ratings ',
                        color_discrete_sequence=my_col,
                        labels= {'count':'Frequiency of the word',
                                 'index':'Words'})
        fig.update_layout(
            height = 600,
            title = { 'x' : 0.4}
        )
        
        return fig
