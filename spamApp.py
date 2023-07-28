import streamlit as st
import pickle
import string
import random
import numpy as np
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    all_words = set()
    spam_words = set()
    for i in text:
        if i.isalnum():
            all_words.add(i)

    for i in all_words:
        if i not in stopwords.words('english') and i not in string.punctuation:
            stemmed_word = ps.stem(i)
            if model.predict(tfidf.transform([stemmed_word]))[0] == 1:
                spam_words.add(stemmed_word)

    return " ".join(all_words), " ".join(spam_words)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Custom spam alert messages
spam_alert_messages = [
    "Oopsie! Our anti-spam radar caught something fishy! 🐠 This message is marked as spam!",
    "Caution! We detected a spammy intruder. Please don't feed the spambot! 🚫",
    "Red alert! This message contains potential spam. We've got it under control! 🚀",
]

# Sidebar
st.sidebar.title("About this App")
st.sidebar.write("This is a simple spam classifier app.")
st.sidebar.write("Enter a message in the text area and click 'Predict' to see if it's spam or not.")
st.sidebar.header("Sample Text")
st.sidebar.write("Hi, congratulations! You have won a $1000 gift card. Claim it now!")
st.sidebar.write("Want to try more examples? Feel free to customize your own text.")

# Main content
st.title("Spam Classifier")

input_text = st.text_area("Enter your message here:")
if st.button("Predict"):
    # 1. Preprocess
    all_words, spam_words = transform_text(input_text)
    # 2. Vectorize
    vector_input = tfidf.transform([spam_words])
    result = model.predict(vector_input)[0]

    # Output
    if result == 1:
        st.header("Spam")
        random_spam_alert = random.choice(spam_alert_messages)
        st.markdown(f'<div style="background-color: #FF0000; padding: 10px; border-radius: 5px; color: white;">{random_spam_alert}</div>', unsafe_allow_html=True)
        # Generate and display word cloud
        st.subheader("Word Cloud (Spam Words)")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()
    else:
        st.header("Not Spam")
        st.success("No worries! Your message is safe.")

# Disable the warning about global usage of pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)
