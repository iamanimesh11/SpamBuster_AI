import time

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
from PIL import  Image

from collections import Counter
st.set_page_config(page_title="spam Buster AI", layout="wide", initial_sidebar_state="expanded",page_icon="💬")

hide_github_icon_js = """
    <style>
    #MainMenu {
        display: none;
    }
    button.css-ch5dnh {
        display: none;
    }
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const toolbar = document.querySelector('[data-testid="stToolbar"]');
        if (toolbar) {
            toolbar.style.display = 'none';
        }
    });
    </script>
    """
st.markdown(hide_github_icon_js, unsafe_allow_html=True)



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
    "Oopsie! Our anti-spam radar caught something fishy! 🐠 ",
    "Caution! We detected a spammy intruder 🚫",
    "Red alert! This message contains potential spam⚠️",
]



linkedin_profile = "https://www.linkedin.com/in/animesh-singh11/"
github_profile = "https://github.com/iamanimesh11"
gmail_address = "iamanimesh11june@gmail.com"


def image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
linkedin_icon_data = image_to_base64(Image.open("icons/linkedin.png"))
github_icon_data = image_to_base64(Image.open("icons/github.png"))
gmail_icon_data = image_to_base64(Image.open("icons/gmail.png"))

icon_size = "50px"
icon_margin = "10px"

css_style = '''
            .icon:hover {
                transform: scale(1.2);
            }
        '''

linkedin_icon = f'<a href="{linkedin_profile}" target="_blank"><img src="data:image/png;base64,{linkedin_icon_data}" class="icon" width="{icon_size}" height="{icon_size}" style="margin-bottom:{icon_margin};"></a>'
github_icon = f'<a href="{github_profile}" target="_blank"><img src="data:image/png;base64,{github_icon_data}" class= "icon" width="{icon_size}" height="{icon_size}"style= border-radius:10px;background-color:white;"margin-bottom:{icon_margin};"></a>'
gmail_icon = f'<a href="mailto:{gmail_address}" target="_blank"><img src="data:image/png;base64,{gmail_icon_data}" class= "icon" width="{icon_size}" height="{icon_size}"style="margin-bottom:{icon_margin};"></a>'

st.markdown(f'<style>{css_style}</style>', unsafe_allow_html=True)

st.sidebar.markdown(   f'<div style="display: flex; justify-content: centre;">{linkedin_icon}<div style="margin-right: 10px;"></div>{github_icon}<div style="margin-right: 10px;"></div>{gmail_icon}</div>',
    unsafe_allow_html=True)
st.sidebar.title("About this App")
navigation_options = ["Home", "Project Overview"]
selected_option = st.sidebar.selectbox("Go to", navigation_options)
st.sidebar.write("Welcome to the Spam Buster AI App!")
st.sidebar.write(" Detect spam messages instantly and stay safe from harmful content. Input your message via text or upload a file. The app will provide a verdict on spam presence and raise a red alert if necessary. Explore the word cloud to see frequent spam-related words. Enjoy the interactive experience and learn about spam classification with this educational app!")
st.sidebar.write('<span style="color: red;">⚠️Please note that while the model is designed to be accurate, no classifier is perfect, and occasional misclassifications may occur.</span>', unsafe_allow_html=True)
st.sidebar.write("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--Animesh")


def show_home_page():

    # Main content
    st.title("Spam Buster AI")
    input_option = st.selectbox("Input Option", ("Type your message", "Upload a file"))
    st.markdown(
        "Made by Animesh |[ website](https://animesh11portfolio.streamlit.app/)| [ LinkedIn](https://www.linkedin.com/in/animesh-singh11)")
    st.toast("Hi whats up",  icon="👋")
    hint_examples = {
        "Hint 1": "Hurry! Limited-time offer: 90% off on all products",
        "Hint 2": "Subject: **CLAIM YOUR EXCLUSIVE PRIZE NOW!**Dear [Recipient],CONGRATULATIONS! 🎉🎉🎉 You are one of the lucky few selected to receive a mind-blowing prize worth thousands of dollars! 🏆💰Our records show that you've won the GRAND PRIZE in our annual giveaway extravaganza! 🎁🎈 This is not a joke or a drill – it's real, and it's all yours for the taking!🎯 Your Exclusive Grand Prize Package Includes:✔️ The latest cutting-edge SMARTPHONE – the envy of all your friends! 📱🤩✔️ A LUXURY VACATION for two to a breathtaking tropical paradise! ✔️ A STATE-OF-THE-ART LAPTOP for peak productivity and entertainment! 💻🎮✔️ A GIFT CARD worth $500 to splurge on your heart's desire! 💳💫BUT WAIT, THERE'S MORE! 🎁🎁🎁 As a special bonus, you'll also get access to our EXCLUSIVE MEMBERSHIP CLUB, where you'll unlock SECRET DEALS and UNBEATABLE DISCOUNTS on top brands!To claim your prize, simply click on the link below or reply to this email with your details, and we'll rush your prize package to your doorstep! 🏃‍♂️[Malicious link or fake email address]ACT NOW! This extraordinary offer is timesensitive and won't wait for anyone! Seize this opportunity of a lifetime and indulge in the ultimate luxury experience! 🌟💎 WARNING: Failure to claim your prize within 48 hours will forfeit your eligibility, and we'll have to award the prize to another lucky winner! 🚨 Don't let that happen!Remember, this is a limited-time chance to make your dreams come true! Don't let it slip away like sand through your fingers! 🏝️⏳Best regards, Prize Patrol Team",
        "Hint 3": "Dear [Recipient], We're thrilled to inform you that you've won a fabulous prize package! Claim your reward now to enjoy an all-inclusive vacation to a stunning beach destination, where you'll be pampered with luxurious accommodations and breathtaking ocean views. Don't miss this once-in-a-lifetime opportunity; click the link below to unlock your dream vacation experience!",
    }
    if input_option == "Type your message":
        input_text = st.text_area("Type your message here...")
        hint_button = st.button("Hint")
        hint_text_placeholder = st.empty()

        if hint_button:
            # Show the example text when the hint button is clicked
            hint_text = random.choice(list(hint_examples.values()))
            hint_text_placeholder.text(hint_text)
        elif hint_text_placeholder.text:
            # Clear the example text when the button is clicked again
            hint_text_placeholder.empty()
    else:
        uploaded_file = st.file_uploader("Upload a file", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode()
    if st.button("Run Spam Check..."):
        with st.spinner("ummm..."):
            # 1. Preprocess
            all_words, spam_words = transform_text(input_text)
            # 2. Vectorize

            time.sleep(1)
            vector_input = tfidf.transform([spam_words])
            result = model.predict(vector_input)[0]


            if result == 1:

                    random_spam_alert = random.choice(spam_alert_messages)
                    st.markdown(f'<div style="background-color: #FF0000; padding: 10px; border-radius: 5px; color: white;">{random_spam_alert}</div>', unsafe_allow_html=True)
                    # Generate and display word cloud
                    st.subheader("Visualize Spam Keywords")
                    # Calculate word frequencies
                    word_freq = Counter(spam_words.split())
                    max_freq = max(word_freq.values())
                    scaled_word_freq = {word: (freq / max_freq) * 40 + 10 for word, freq in word_freq.items()}
                    wordcloud = WordCloud(width=400, height=200, background_color='black', colormap='plasma',random_state=40,max_font_size=80, min_font_size=10).generate_from_frequencies(scaled_word_freq)
                    plt.figure(figsize=(5, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot()

                    st.toast("hope,you liked the results", icon="😊")
                    time.sleep(12)
            else:
                st.balloons()
                st.success("No worries!😊 Your message is safe.")

                st.toast("hope,you liked the results", icon="😊")
                time.sleep(5)


def show_project_overview_page():
    st.title("Introduction to the Spam Classifier Project")

    st.markdown("""
        The Spam Classifier project aims to build a machine learning model that can effectively classify text messages as either "spam" or "ham" (non-spam). The dataset used for training and evaluation contains labeled text messages, where "spam" represents unwanted promotional or unsolicited messages, and "ham" denotes regular messages.
        """)

    st.title("Project Overview")

    st.markdown("""
        The project is structured into several key steps:
        """)

    st.subheader("1. Data Cleaning and Exploratory Data Analysis (EDA):")
    st.markdown("""
        - The dataset is loaded and checked for missing values and duplicates.
        - Unnecessary columns are dropped from the dataset.
        - The target column is encoded to convert "spam" and "ham" labels into numerical values (0 and 1).
        - EDA is performed to understand the distribution of spam and ham messages, the length of messages, and other useful insights.
        """)

    st.subheader("2. Data Preprocessing:")
    st.markdown("""
        - Text data is preprocessed to prepare it for modeling.
        - All text is converted to lowercase for consistency.
        - Tokenization is performed to break down sentences into words.
        - Special characters and punctuation are removed.
        - Stop words (commonly occurring words like "a," "the," etc.) are eliminated to reduce noise.
        - Words are stemmed to their root form, improving model efficiency.
        """)

    st.subheader("3. Model Building:")
    st.markdown("""
        - The preprocessed text data is vectorized using the Term Frequency-Inverse Document Frequency (TF-IDF) technique.
        - Various classification algorithms are explored, including Naive Bayes (Gaussian, Multinomial, and Bernoulli), Logistic Regression, Support Vector Machines, Decision Trees, K-Nearest Neighbors, Random Forests, AdaBoost, and Extra Trees.
        - The models are trained and evaluated using accuracy and precision metrics.
        """)

    st.subheader("4. Model Improvement:")
    st.markdown("""
        - Different techniques are employed to enhance model performance.
        - Feature selection and hyperparameter tuning are considered.
        - Ensembling methods like Voting Classifier and Stacking Classifier are experimented with to combine the strengths of different models.
        """)

    st.subheader("5. Model Deployment:")
    st.markdown("""
        - The best-performing model is selected and saved for future use.
        - The TF-IDF vectorizer is also saved to preprocess incoming text messages for classification.
        """)

    st.title("Future Work")
    st.markdown("""
    While the current version of the Spam Classifier project is already providing valuable insights and accurate predictions, there are several exciting possibilities for further enhancement:
    """)

    st.subheader("Expand Model Selection:")
    st.markdown("""
    We aim to incorporate more state-of-the-art machine learning models into the Spam Classifier. By exploring a broader range of algorithms and ensembling techniques, we can offer users even better accuracy and reliability in classifying text messages.
    """)

    st.subheader("User Data History:")
    st.markdown("""
    To enhance user experience, we plan to implement a feature that allows users to save their data history and access it anytime they need. This feature will enable users to keep track of their previous classifications and revisit them for future reference.
    """)

    st.subheader("Text Highlighting Error Analysis:")
    st.markdown("""
    We are working on integrating a text highlighting tool into the Spam Classifier. This tool will analyze misclassified text messages and highlight specific words or phrases that influenced the incorrect prediction. Users can use this feature to gain insights into why certain messages were misclassified, thereby improving the model's performance.
    """)

    st.subheader("Dynamic Training with Fresh Data:")
    st.markdown("""
    To ensure the Spam Classifier remains up-to-date and adaptive, we plan to introduce periodic training updates using new and diverse data. This process will help the model adjust to changing spam patterns and continue delivering accurate results.
    """)

    st.subheader("Customizable Thresholds:")
    st.markdown("""
    In the future, we intend to allow users to adjust the classification thresholds according to their preferences. This feature will provide users with more control over the trade-off between precision and recall, catering to individual requirements.
    """)

    st.subheader("Multilingual Support:")
    st.markdown("""
    Expanding the Spam Classifier's capabilities to support multiple languages is on our radar. By incorporating multilingual data and NLP techniques, we aim to make the model more versatile and accessible to a broader audience.
    """)

    st.subheader("Real-Time Classification:")
    st.markdown("""
    To enhance user convenience, we are exploring options for real-time classification. Users will be able to submit messages instantly and receive prompt results, making the app more user-friendly.
    """)

    st.subheader("User Feedback Loop:")
    st.markdown("""
    We value user feedback and plan to establish a user feedback loop for continuous improvement. User suggestions and reports of misclassified messages will help us fine-tune the model and optimize its performance.
    """)

if selected_option == "Home":
    show_home_page()
elif selected_option == "Project Overview":
    show_project_overview_page()





    # Disable the warning about global usage of pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)
