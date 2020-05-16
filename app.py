import pandas as pd
import nltk 
import numpy as np
import re
from nltk.stem import wordnet # to perform lemmitization
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.feature_extraction.text import TfidfVectorizer # to perform tfidf
from nltk import pos_tag # for parts of speech
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity
from nltk import word_tokenize # to create tokens
from nltk.corpus import stopwords # for stop words
import random

from flask import Flask, render_template, request
app = Flask(__name__)

df=pd.read_csv('https://raw.githubusercontent.com/jytg17/prism/master/dialog_talk_agent.csv')
df.ffill(axis = 0,inplace=True) # fills the null value with the previous value.

# function that performs text normalization steps

def text_normalization(text):
    text=str(text).lower() # text to lower case
    spl_char_text=re.sub(r'[^ a-z]','',text) # removing special characters
    tokens=nltk.word_tokenize(spl_char_text) # word tokenizing
    lema=wordnet.WordNetLemmatizer() # intializing lemmatization
    tags_list=pos_tag(tokens,tagset=None) # parts of speech
    lema_words=[]   # empty list 
    for token,pos_token in tags_list:
        if pos_token.startswith('V'):  # Verb
            pos_val='v'
        elif pos_token.startswith('J'): # Adjective
            pos_val='a'
        elif pos_token.startswith('R'): # Adverb
            pos_val='r'
        else:
            pos_val='n' # Noun
        lema_token=lema.lemmatize(token,pos_val) # performing lemmatization
        lema_words.append(lema_token) # appending the lemmatized token into a list
    
    return " ".join(lema_words) # returns the lemmatized tokens as a sentence 

df['lemmatized_text']=df['Context'].apply(text_normalization) # applying the fuction to the dataset to get clean text

# all the stop words we have 

stop = stopwords.words('english')

cv = CountVectorizer() # intializing the count vectorizer
X = cv.fit_transform(df['lemmatized_text']).toarray()

# returns all the unique word from data 

features = cv.get_feature_names()
df_bow = pd.DataFrame(X, columns = features)
df_bow.head()

# # Model Using Bag of Words

# Function that removes stop words and process the text

def stopword_(text):   
    tag_list=pos_tag(nltk.word_tokenize(text),tagset=None)
    stop=stopwords.words('english')
    lema=wordnet.WordNetLemmatizer()
    lema_word=[]
    for token,pos_token in tag_list:
        if token in stop:
            continue
        if pos_token.startswith('V'):
            pos_val='v'
        elif pos_token.startswith('J'):
            pos_val='a'
        elif pos_token.startswith('R'):
            pos_val='r'
        else:
            pos_val='n'
        lema_token=lema.lemmatize(token,pos_val)
        lema_word.append(lema_token)
    return " ".join(lema_word) 

# defining a function that returns response to query using bow

def chat_bow(text):
    s=stopword_(text)
    lemma=text_normalization(s) # calling the function to perform text normalization
    bow=cv.transform([lemma]).toarray() # applying bow
    cosine_value = 1- pairwise_distances(df_bow,bow, metric = 'cosine' )
    index_value=cosine_value.argmax() # getting index value 
    return df['Text Response'].loc[index_value]


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey", "yo")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

abbr = {'amk': 'ang mo kio',
 'bd': 'bedok',
 'bi': 'bishan',
 'bb': 'bukit batok',
 'bm': 'bukit merah',
 'bp': 'bukit panjang',
 'cck': 'choa chu kang',
 'cl': 'clementi',
 'gl': 'geylang',
 'hg': 'hougang',
 'je': 'jurong east',
 'jw': 'jurong west',
 'kw': 'kallang whampoa',
 'pr': 'pasir ris',
 'pg': 'punggol',
 'qt': 'queesntown',
 'sb': 'sembawang',
 'sk': 'sengkang',
 'sg': 'serangoon',
 'tam': 'tampines',
 'tg': 'tegah',
 'tpy': 'toa payoh',
 'wd': 'woodlands',
 'ys': 'yishun',
 'du': 'dwelling units',
 'dus': 'dwelling units'}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def gazer():
    text = request.args.get('msg')
    #flag=True
    #print("ROBO: My name is Robo. I will answer your queries about HDB towns. If you want to exit, type Bye!")
    #while(flag==True):
    text=text.lower()
    for key in abbr.keys():
        text=text.replace(key, abbr[key])
    ans = ''
    if(text!='bye'):
        if(text=='thanks' or text=='thank you'):
            #flag=False
            ans = ("PRISM: You are welcome..")
        else:
            if(greeting(text)!=None):
                ans = ("PRISM: "+ greeting(text))
            else:
                #print("ROBO: ",end="")
                ans = ("PRISM: " + chat_bow(text))
                
    else:
        #flag=False
        ans = ("PRISM: Bye! take care..")
    
    return str(ans)

if __name__ == "__main__":
    app.run()
