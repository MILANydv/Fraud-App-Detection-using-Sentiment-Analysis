import hashlib
import pickle
import re
import sqlite3
from io import StringIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

st.title("Predicting the sentiment of a review")


# Security
# passlib,hashlib,bcrypt,scrypt


def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',
              (username, password))
    conn.commit()


def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',
              (username, password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


def main():
    """Simple Login App"""

    menu = ["Home", "Login", "SignUp"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        components.html(
            "<html><body><p>Most of us use Android and iOS phones these days, and we frequently utilize the play store or app store. Both marketplaces have a large number of applications, but unfortunately, some of them are fraudulent. Such programs might cause phone harm as well as data theft. As a result, such programs must be labeled so that store patrons can identify them. As a result, we propose a web application that will handle the information, comments, and application evaluation. As a result, determining whether application is fraudulent will be much easy. With the online application, many applications may be handled at once. Furthermore, the user may not always find accurate or authentic product reviews on the internet.</p> <img src='https://miro.medium.com/max/626/0*pfvAPHKzpyFiYO3U.jpg'</body></html>", width=600, height=800)
    elif choice == "Login":

        st.subheader("Login Section")
        username = st.text_input("User Name")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username, check_hashes(password, hashed_pswd))
            if result:

                st.success("Logged In as {}".format(username))
                uploaded_file = st.file_uploader("Choose a file")
                if uploaded_file is not None:
                    # To read file as bytes:
                    bytes_data = uploaded_file.getvalue()

                # To convert to a string based IO:
                    stringio = StringIO(
                        uploaded_file.getvalue().decode("utf-8"))

                    # To read file as string:
                    string_data = stringio.read()

                    # Can be used wherever a "file-like" object is accepted:
                    dataframe = pd.read_csv(uploaded_file)

                    all_stopwords = stopwords.words('english')
                    all_stopwords.remove('not')

                    # load pickled model
                    cvFile = './data/model/c1_BoW_Sentiment_Model.pkl'
                    cv = pickle.load(open(cvFile, "rb"))
                    corpus = []

                    for i in range(0, dataframe.shape[0]):
                        review = re.sub('[^a-zA-Z]', ' ', dataframe['Sentence'][i])
                        review = review.lower()
                        review = review.split()
                        review = [ps.stem(word)
                                for word in review if not word in set(all_stopwords)]
                        review = ' '.join(review)
                        corpus.append(review)
                        X_fresh = cv.transform(corpus).toarray()

                    # loading model
                    classifier = joblib.load(
                        './data/model/c2_Classifier_Sentiment_Model')
                    y_pred = classifier.predict(X_fresh)

                    dataframe['predicted_label'] = pd.Series(y_pred)

                    if dataframe.to_csv(
                            "./data/predicated data/Predicted_Sentiments_Fresh_Dump.csv") is not None:
                        st.success("File uploaded successfully")
                    else:
                        st.table(dataframe)

            else:
                st.warning("Incorrect Username/Password")

    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password", type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user, make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")


if __name__ == '__main__':
    main()

# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     # To read file as bytes:
#     bytes_data = uploaded_file.getvalue()

#     # To convert to a string based IO:
#     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

#     # To read file as string:
#     string_data = stringio.read()

#     # Can be used wherever a "file-like" object is accepted:
#     dataframe = pd.read_csv(uploaded_file)

#     all_stopwords = stopwords.words('english')
#     all_stopwords.remove('not')

#     # load pickled model
#     cvFile = './data/model/c1_BoW_Sentiment_Model.pkl'
#     cv = pickle.load(open(cvFile, "rb"))
#     corpus = []

#     for i in range(0, dataframe.shape[0]):
#         review = re.sub('[^a-zA-Z]', ' ', dataframe['Sentence'][i])
#         review = review.lower()
#         review = review.split()
#         review = [ps.stem(word)
#                   for word in review if not word in set(all_stopwords)]
#         review = ' '.join(review)
#         corpus.append(review)
#     X_fresh = cv.transform(corpus).toarray()

#     # loading model
#     classifier = joblib.load('./data/model/c2_Classifier_Sentiment_Model')
#     y_pred = classifier.predict(X_fresh)

#     dataframe['predicted_label'] = pd.Series(y_pred)

#     if dataframe.to_csv(
#             "./data/predicated data/Predicted_Sentiments_Fresh_Dump.csv") is not None:
#         st.success("File uploaded successfully")
#     else:
#         st.table(dataframe)
