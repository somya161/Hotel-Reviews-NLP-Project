#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:22:17 2022

@author: somya
"""



# Importing the libraries
import streamlit as st
import pickle
import spacy
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


# Body of the application
st.header("Hotel Review Prediction Application.")
st.markdown("This application is trained on machine learning model.\n "
            "This application can predict if the given **review**"
            " is **Positive, Negative or Neutral**")


text = st.text_input("Type your review here and click on Result...", """""")


# Working on prediction
# Loading trained model
loaded_model = pickle.load(
    open("model_pipe.pkl", "rb"))


# Preprocessing the text
nlp = spacy.load("en_core_web_sm")

def preprocessing(text):
    """Takes the text input and removes the stop words and punctuations from the text and gives processed text output.
    """
    global nlp
    doc = nlp(text)

    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)


preprocessed_input = [preprocessing(text)]
#st.write(preprocessed_input)

# Tfidf


# Making prediction
if st.button("Result"):

    # Making prediction for model input
    prediction = loaded_model.predict(preprocessed_input)
    st.write(prediction)
 # Returning true prediction
    if prediction == 'Negative':
        st.write("Input review has Negative Sentiment:sad:🙁")
    elif prediction == 'Positive':
        st.write("Input review has Positive Sentiment:smile:😀")
    else:
        st.write("Input review has Neutral Sentiment😐")

 