import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nbformat
import re
import emoji
import nltk
from nltk.tokenize import sent_tokenize
import contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import langid
import warnings
warnings.filterwarnings("ignore")

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('vader_lexicon')

lemmatizer =  WordNetLemmatizer()
stopwords = stopwords.words('english')
analyzer = SentimentIntensityAnalyzer()

my_col = [
    '#FF6F61',  
    '#6B5B95',  
    '#88B04B', 
    '#F7CAC9', 
    '#92A8D1',  
    '#FFA500', 
    '#00CED1', 
    '#FFB347',  
    '#D65076',  
    '#45B8AC'   
]
class chatgpt_review():
    def __init__(self):
        pass

    def clean(self,df):
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df['language'] = df['content'].apply(lambda x : langid.classify(x)[0])
        df = df[df['language'] == 'en']
        df.reset_index(drop=True,inplace=True)
        df.drop(columns=['reviewCreatedVersion'],inplace=True)
        df = df[['at','appVersion','userName','content','thumbsUpCount','score']]
        df.columns = ['date','version','user_name','review','thumbs_up_count','rating']
        # Text Cleaning
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
            # Remove Emojies
            text = emoji.replace_emoji(text,replace='')
            # Remove White sapace
            text = re.sub(r'\s+', ' ', text).strip()

            return text
        # lemmatation
        def lemma(text):
            text = word_tokenize(text)
            text =  [contractions.fix(word) for word in text]
            text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords]
            text = ' '.join(text)
            return text
        
        def label(text):
            score = analyzer.polarity_scores(text)
            compound = score['compound']
            
            if compound >= 0.05:
                sentiment = 'pos'
            elif compound <= -0.05:
                sentiment = 'neg'
            else:
                sentiment = 'neu'
            
            return sentiment

        df['review'] = df['review'].apply(text_cleaning)
        df = df[df['review'] != ''] # Filter out empty spaces
        df.reset_index(drop=True,inplace=True)
        df['label'] = df['review'].apply(label)
        df['final_text'] = df['review'].apply(lemma)
        df['char_len'] = df['final_text'].apply(lambda x:len(x))
        df['word_len'] = df['final_text'].apply(lambda x : len(x.split()))
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        df = df[['date', 'version', 'user_name', 'review', 'thumbs_up_count', 'rating',
        'final_text', 'char_len', 'word_len', 'label']]

        return df
    
    def lab_dis(self,df):
        fig = px.histogram(df,
                        x ='label',
                        color = 'label',
                        color_discrete_sequence = my_col,
                        title='Sentiment Distribution'
                            )
        fig.update_layout(
                        title ={'x':0.25}
                        )

        return fig
    
    def rat_dis(self,df):
        fig = px.histogram(df,
                           x='rating',
                           color='label',
                           color_discrete_sequence=my_col,
                           title='Rating Distribution by Rating',
                           )
        fig.update_layout(
                        title ={'x':0.25}
                        )

        return fig
    
    def rat_pie(self,df):
        fig = px.pie(df.groupby('rating').count()['final_text'],
         values='final_text',
         names=df.groupby('rating').count().index,
         title='Rating Distribution Pie Chart'
        )
        fig.update_layout(
                        title ={'x':0.25}
                        )
        
        return fig 
    
    def char_dis(self,df):
        fig = px.histogram(df,
                           x='char_len',
                           color='label',
                           color_discrete_sequence=my_col,
                           title='Character Length Distribution by Sentiment')
        fig.update_layout(
                        title ={'x':0.25}
                        )
        return fig
    
    def word_dis(self,df):
        fig = px.histogram(df,
                           x='word_len',
                           color='label',
                           color_discrete_sequence=my_col,
                           title='Word Length Distribution by Sentiment')
        fig.update_layout(
                        title ={'x':0.25}
                        )
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
    
    def rat_comp(self,df):
        fig = px.imshow(pd.crosstab(df['rating'],df['label']),
                  text_auto=True,
                  title='Rating vs Sentiment Heatmap')
        fig.update_layout(
                title ={'x':0.5})
        
        return fig
    
    def top_users(self,df):
        t_df = df['user_name'].value_counts().head(10).reset_index()
        fig = px.bar(t_df,
                    x='user_name',
                    y='count',
                    color='user_name',
                    title='Top 10 Active Users by Reviews'
                    )
        fig.update_layout(
                        title ={'x':0.25},
                        )
        fig.update_traces(showlegend=False)

        return fig
    
    def review_over_year(self,df):
        t_df = df['date'].dt.year.value_counts().reset_index()
        fig = px.bar(t_df,
                    x='date',
                    y='count',
                    color='date',
                    title='Reviews by Year',
                    color_discrete_sequence=my_col
                    )
        fig.update_layout(
                                title ={'x':0.25},
                                )
        fig.update_traces(showlegend=False)

        return fig
    
    def review_over_month(self,df):
        t_df = df
        t_df['month'] = t_df['date'].dt.month_name()
        fig = px.imshow(pd.crosstab(t_df['label'],t_df['month']),
                text_auto=True,title='Sentiment vs Month Heatmap')
        fig.update_layout(
            width = 800
        )
        

        return fig
    
    def char_average(self,df):
        t_df = df.groupby('label')['char_len'].mean().reset_index()
        fig = px.bar(t_df,
               x='label',
               y='char_len',
               color='label',
               color_discrete_sequence=my_col,
               title='Average Character Length by Sentiment'
                )
        fig.update_layout(
                        title ={'x':0.25},
                        )
      
        return fig
    
    def word_average(self,df):
        t_df = df.groupby('label')['word_len'].mean().reset_index()
        fig = px.bar(t_df,
                x='label',
                y='word_len',
                color='label',
                color_discrete_sequence=my_col,
                title='Average Word Length by Sentiment'
                )
        fig.update_layout(
                        title ={'x':0.25},
                        )
      
        return fig

    def frq_count_over_rating(self,df):
        t_df1 = df[df['rating'] == 1]['final_text'].str.split().explode().value_counts().head(10).reset_index()
        t_df1['rating'] = 1
        t_df2 = df[df['rating'] == 2]['final_text'].str.split().explode().value_counts().head(10).reset_index()
        t_df2['rating'] = 2
        t_df3 = df[df['rating'] == 3]['final_text'].str.split().explode().value_counts().head(10).reset_index()
        t_df3['rating'] = 3
        t_df4 = df[df['rating'] == 4]['final_text'].str.split().explode().value_counts().head(10).reset_index()
        t_df4['rating'] = 4
        t_df5 = df[df['rating'] == 5]['final_text'].str.split().explode().value_counts().head(10).reset_index()
        t_df5['rating'] = 5

        t_df = pd.concat([t_df1,t_df2,t_df3,t_df4,t_df5],axis=0)

        t_df

        fig = px.funnel(t_df,
                        y = 'count',
                        x = 'final_text',
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
    
    def frq_count_over_label(self,df):
        t_df1 = df[df['label'] == 'pos']['final_text'].str.split().explode().value_counts().head(10).reset_index()
        t_df1['rating'] = 'pos'
        t_df2 = df[df['label'] == 'neu']['final_text'].str.split().explode().value_counts().head(10).reset_index()
        t_df2['rating'] = 'neu'
        t_df3 = df[df['label'] == 'neg']['final_text'].str.split().explode().value_counts().head(10).reset_index()
        t_df3['rating'] = 'neg'

        t_df = pd.concat([t_df1,t_df2,t_df3],axis=0)

        
        fig = px.funnel(t_df,
                        y = 'count',
                        x = 'final_text',
                        color='rating',
                        title = 'Top 10 Repeting words in each label ',
                        color_discrete_sequence=my_col,
                        labels= {'count':'Frequiency of the word',
                                    'index':'Words'})
        fig.update_layout(
            height = 600,
            title = { 'x' : 0.4}
                )
        
        return fig
    
    def overall_wordcloud(self,df):
        pos_words = df[df['label'] == 'pos']['final_text'].str.split().explode().to_list()
        pos_words = ' '.join([str(word) for word in pos_words])
        wordcloud=WordCloud(width=800,height=500,random_state=42,max_font_size=110).generate(pos_words)
        fig1 = px.imshow(wordcloud,title="Positive Words")

        fig1.update_layout(
            title ={'x':0.3}
        )
        fig1.update_xaxes(visible=False)
        fig1.update_yaxes(visible=False)

        neu_words = df[df['label'] == 'neu']['final_text'].str.split().explode().to_list()
        neu_words = ' '.join([str(word) for word in neu_words])
        wordcloud=WordCloud(width=800,height=500,random_state=42,max_font_size=110).generate(neu_words)
        fig2 = px.imshow(wordcloud,title="Neutral Words")

        fig2.update_layout(
            title ={'x':0.3}
        )
        fig2.update_xaxes(visible=False)
        fig2.update_yaxes(visible=False)

        neg_words = df[df['label'] == 'neg']['final_text'].str.split().explode().to_list()
        neg_words = ' '.join([str(word) for word in neg_words])
        wordcloud=WordCloud(width=800,height=500,random_state=42,max_font_size=110).generate(neg_words)
        fig3 = px.imshow(wordcloud,title="Negative Words")

        fig3.update_layout(
            title ={'x':0.3}
        )
        fig3.update_xaxes(visible=False)
        fig3.update_yaxes(visible=False)

        return fig1, fig2, fig3
    
    def thump_up(self,df):
        t_df = df.groupby('label')['thumbs_up_count'].sum().reset_index()
        fig = px.pie(
            t_df,
            values='thumbs_up_count',
            hover_name='label',
            hole=0.5
            )
        fig.update_layout(
            title ={'x':0.3}
        )

        return fig
        