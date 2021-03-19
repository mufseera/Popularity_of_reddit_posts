from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the model from disk
filename = 'rfmodel.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
	    message = request.form['Title']
	    data = [message]
	    vect = cv.transform(data).toarray()
	    newtext=pd.DataFrame(vect)
	    Upvote_ratio= float(request.form['Upvote_ratio'])
	    Gilded = request.form["Gilded"]
	    Number_of_Comments = int(request.form['Number_of_Comments'])
	    neg = float(request.form['neg'])
	    neu = float(request.form['neu'])
	    pos= float(request.form['pos'])
	    compound= float(request.form['compound'])
	    data = np.array([[ Upvote_ratio, Gilded, neg, Number_of_Comments, neu, pos, compound]])
	    data1=pd.DataFrame(data)
	    newdata=pd.concat([data1,newtext],axis=1)
	    prediction=clf.predict(newdata)
	    output=round(prediction[0],2)
	    return render_template('home.html',prediction_text="Score of your Title is. {}".format(output))
	    return render_template("home.html")




	    my_prediction =clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)


