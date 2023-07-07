import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()
nltk.download('stopwords')
vector_form = pickle.load(open('vect.pkl', 'rb'))
load_model = pickle.load(open('fake_news.pkl', 'rb'))


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction





if __name__ == '__main__':
    # Set page configuration
    # Set page configuration
    st.set_page_config(
        page_title="समाचार.AI",
        page_icon="📰",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Add some custom CSS styles
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #121212;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 0.25rem;
            cursor: pointer;
        }

        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #000000;
            border-radius: 0.25rem;
            padding: 0.5rem;
            border: 1px solid #ced4da;
        }

        .stTextInput>div>div>input:focus {
            outline: none;
            box-shadow: 0 0 0 0.25rem rgba(76, 175, 80, 0.25);
        }

        body {
            background-image: url('1.jpg');
            background-repeat: no-repeat;
            background-size: cover;
            background-position: center center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # App title

    st.title("समाचार.AI  🕵️📰")
    st.title("Fake News Detector Web App")

    # App content
    with st.container():
        st.markdown("Enter a news article below to detect if it's fake or real.")
        text_input = st.text_area("News Article", height=150)
        predict_btt = st.button("Detect")

        if predict_btt:
            # Perform the fake news detection logic
            # You can add your code here to process the input

            # Display the results
            prediction_class = fake_news(text_input)
            print(prediction_class)
            if prediction_class == [0]:
                st.success('Reliable')
            if prediction_class == [1]:
                st.warning('Unreliable')


