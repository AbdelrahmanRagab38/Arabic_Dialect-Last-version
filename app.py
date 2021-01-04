# Load Packages

from flask import Flask, render_template, url_for, request
#import textblob
#from textblob import TextBlob
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
# Create our Flask Object Instance

app = Flask(__name__)

# Set Home Page Template

@app.route('/')
def home():
	return render_template('home.html')

# Set Predict Page
# Using TextBlob package (powered by the Google Translate API)

@app.route('/predict',methods=['GET', 'POST'])

def predict():
    if request.method == 'POST':
        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        data2 = pd.read_csv('Arabic_Data_cleaned_without_duplicated.csv')
        text2 = data2['Arabic_Tweets_Cleaned'].values.tolist()
        target2 = data2['labels_new'].values.tolist()
        X_train2, X_test2, y_train2, y_test2 = train_test_split(text2, target2, test_size=0.2, shuffle=True,random_state=42)
        X_train_counts2 = count_vect.fit_transform(X_train2)
        X_tfidf_train2 = tfidf_transformer.fit_transform(X_train_counts2)
        X_test_counts2 = count_vect.transform(X_test2)
        X_tfidf_test2 = tfidf_transformer.transform(X_test_counts2)
        clf = MultinomialNB().fit(X_tfidf_train2, y_train2)


        message = request.form['message']
       # message = str(message)
        #print(str(message))
        message = [message]
        X_testing_counts = count_vect.transform(message)
        X_tfidf_testing = tfidf_transformer.transform(X_testing_counts)
        detect = clf.predict(X_tfidf_testing)
        print(detect)
        #blobline = TextBlob(message)
        #detect = blobline.detect_language()
    return render_template('result.html',prediction = detect)

if __name__ == '__main__':
	app.run()