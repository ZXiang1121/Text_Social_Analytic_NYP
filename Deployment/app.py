from flask import Flask, render_template, redirect,url_for, request
import pandas as pd
import numpy as np
import pickle
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('stopwords')


app = Flask(__name__)

lr_file = "lr.pkl"

v_file = "vectorizer.pkl"

s_file = "selected_feature.pkl"


def lemmatizing(cleaned_arr):
    lemmatizer = WordNetLemmatizer()
    
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:         
            return None
        
    pos_tagged = nltk.pos_tag(cleaned_arr)
    
    # wordnet_tagged return something like [('wanted', 'v'), ('love', 'n'), ('even', 'r')]
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as it is
            lemmatized_sentence.append(word)
        else:       
            # else use the tag to lemmatize the token to lemmatize the matched tagging with the word
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)

    return lemmatized_sentence

def text_preprocessing(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    lemmatized_text = lemmatizing(words)
    
    return lemmatized_text


@app.route('/')
def home():

    return render_template('home.html')
    
@app.route('/predict', methods=['POST'])
def predict():

    with open(lr_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(v_file, 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open(s_file, 'rb') as f:
        selector = pickle.load(f)

    if request.method == "POST":
        message = request.form['message']
        message = text_preprocessing(message)
        data = [message]
        classes = model.classes_
        new_text_vector = vectorizer.transform(data)
        new_text_vector = selector.transform(new_text_vector)
        predicted_label = (model.predict(new_text_vector))[0]
        y_prob = (model.predict_proba(new_text_vector))
        y_prob = np.round(y_prob * 100, decimals=2).flatten()

    return render_template('result.html', prediction=predicted_label, probability = y_prob, classes=classes)



if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000)