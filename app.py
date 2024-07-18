from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import re
import string

app = Flask(__name__)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
def clean(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('https?://\S+', '', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text

@app.route('/' , methods=['GET','POST'])
def index():
    prediction=None
    news_text=''
    if request.method == 'POST':
        news_text = request.form['news_text']
        cleaned_text = clean(news_text)
        input_vec = vectorizer.transform([cleaned_text])
        prediction = model.predict(input_vec)
        prediction_text = 'True' if prediction[0] == 1 else 'Fake'

        print(f"Input Text: {news_text}")
        print(f"Cleaned Text: {cleaned_text}")
        print(f"Prediction: {prediction_text}")
        return render_template('index.html', prediction=prediction_text, news_text=news_text)
    return render_template('index.html',prediction=prediction,news_text=news_text)
if __name__ == '__main__':
    app.run(debug=True)
