import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
mb = pickle.load(open('model.pkl', 'rb'))

st.title('Spam Classifier')

msg = st.text_area('Enter The Message')



def data_preprocessing(text):
    # lowercase
    text = text.lower()
    # tokenize
    text = nltk.word_tokenize(text)
    y=[]
    # special character remove
    for t in text:
        if t.isalnum():
            y.append(t)
    # stopwords and punctuation removal
    swt=[]
    for t in y:
        if t not in stopwords.words('english') and t not in string.punctuation:
            swt.append(t)
    # stemming
    stt=[]
    for t in swt:
        stt.append(ps.stem(t))
    
    return " ".join(stt)
    
    
if st.button('Predict'):
    # preprocess
    msg = data_preprocessing(msg)
    # vectorise
    msg = tfidf.transform([msg])
    # predict
    result = mb.predict(msg)[0]
    # display
    if result:
        st.header('Spam')
    else:
        st.header('Not Spam')