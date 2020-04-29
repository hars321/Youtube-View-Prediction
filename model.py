import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import matplotlib.pyplot as plt

start_time = time.time()
df = pd.read_csv("data_final.csv")
df = df.sample(frac=1).reset_index(drop=True)


df = df.reset_index()
df.head()

df = df.reset_index()
df.head()


continous_name = ['channel_subscriberCount','likeCount','channelViewCount/socialLink']
continous_name

# Making the training and test set.

X = df[continous_name]
print ("Training Set Shape",X.shape)
Y = df.viewCount
print("Testing Set Shape",Y.shape)

print ("Training in progress.....")

# parameters
n_estimators = 200
max_depth = 25
min_samples_split=15
min_samples_leaf=2


# Random forest classifier
clf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

# It is trained of 2 Epochs
X = np.concatenate((X,X),axis=0)
Y = np.concatenate((Y,Y),axis=0)
clf.fit(X,Y)


print ("Feature Importance ranking",clf.feature_importances_)

model = pickle.load(open('model.pkl','rb'))
e=model.predict([[656024,24885.0,8010063]])
e=int(e)
print("The Number of views in this video will be",e,".")