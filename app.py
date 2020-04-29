import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def new1():
    return "Code is working perfectly"
    
  

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    c=request.get_json()
    a=c["channel_subscriberCount"]
    b=c["likeCount"]
    d=c["channelViewCount/socialLink"]
    int_features=[]
    int_features.append(a)
    int_features.append(b)
    int_features.append(d)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output1 =prediction[0]
    hash1={"output":output1}
    hash1=jsonify(hash1)
    return hash1

if __name__ == "__main__":
    app.run(debug=True)
