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
	    message = request.form['title']
	    data = [message]
	    vect = cv.transform(data).toarray()
	    newtext=pd.DataFrame(vect)
	    upvote_ratio= float(request.form['upvote_ratio'])
	    gilded = request.form["gilded"]
	    num_comments = int(request.form['num_comments'])
	    ups=int(request.form['ups'])
	    data = np.array([[num_comments,gilded,upvote_ratio,ups]])
	    data1=pd.DataFrame(data)
	    newdata=pd.concat([newtext,data1],axis=1)
	    prediction=clf.predict(newdata)
	    output=round(prediction[0],2)
	    return render_template('home.html',prediction_text="Score of your Title is. {}".format(output))
	    return render_template("home.html")




	    my_prediction =clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)


