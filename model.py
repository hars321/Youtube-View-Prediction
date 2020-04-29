{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "import pickle\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "start_time = time.time()\n",
    "df = pd.read_csv(\"data_final.csv\")\n",
    "df = df.sample(frac=1).reset_index(drop=True)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>index</th>\n",
       "      <th>categoryId</th>\n",
       "      <th>channel_subscriberCount</th>\n",
       "      <th>definition</th>\n",
       "      <th>likeCount</th>\n",
       "      <th>dislikeCount</th>\n",
       "      <th>viewCount</th>\n",
       "      <th>commentCount</th>\n",
       "      <th>viewCount/channel_month_old</th>\n",
       "      <th>viewCount/video_month_old</th>\n",
       "      <th>viewCount/http_in_descp</th>\n",
       "      <th>viewCount/NoOfTags</th>\n",
       "      <th>viewCount/tags_in_desc</th>\n",
       "      <th>social_links</th>\n",
       "      <th>subscriberCount/videoCount</th>\n",
       "      <th>channelViewCount/channeVideoCount</th>\n",
       "      <th>channelViewCount/socialLink</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>1252465</td>\n",
       "      <td>0</td>\n",
       "      <td>6681.0</td>\n",
       "      <td>67.0</td>\n",
       "      <td>235437</td>\n",
       "      <td>1214.0</td>\n",
       "      <td>3923.950000</td>\n",
       "      <td>6539</td>\n",
       "      <td>58859</td>\n",
       "      <td>47087</td>\n",
       "      <td>78479</td>\n",
       "      <td>4</td>\n",
       "      <td>3219</td>\n",
       "      <td>329107</td>\n",
       "      <td>25604601</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>328043</td>\n",
       "      <td>0</td>\n",
       "      <td>4001.0</td>\n",
       "      <td>68.0</td>\n",
       "      <td>221239</td>\n",
       "      <td>541.0</td>\n",
       "      <td>6913.718750</td>\n",
       "      <td>18436</td>\n",
       "      <td>31605</td>\n",
       "      <td>5822</td>\n",
       "      <td>13014</td>\n",
       "      <td>7</td>\n",
       "      <td>1333</td>\n",
       "      <td>120817</td>\n",
       "      <td>3715148</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>11</td>\n",
       "      <td>5003475</td>\n",
       "      <td>0</td>\n",
       "      <td>499.0</td>\n",
       "      <td>39.0</td>\n",
       "      <td>200147</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1803.126126</td>\n",
       "      <td>2599</td>\n",
       "      <td>100074</td>\n",
       "      <td>18195</td>\n",
       "      <td>13343</td>\n",
       "      <td>4</td>\n",
       "      <td>227</td>\n",
       "      <td>119078</td>\n",
       "      <td>524541697</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>731237</td>\n",
       "      <td>0</td>\n",
       "      <td>2105.0</td>\n",
       "      <td>53.0</td>\n",
       "      <td>202492</td>\n",
       "      <td>253.0</td>\n",
       "      <td>1965.941748</td>\n",
       "      <td>6982</td>\n",
       "      <td>101246</td>\n",
       "      <td>40498</td>\n",
       "      <td>67497</td>\n",
       "      <td>5</td>\n",
       "      <td>4202</td>\n",
       "      <td>293518</td>\n",
       "      <td>8512049</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>14</td>\n",
       "      <td>5922633</td>\n",
       "      <td>0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4074</td>\n",
       "      <td>2.0</td>\n",
       "      <td>44.769231</td>\n",
       "      <td>140</td>\n",
       "      <td>4075</td>\n",
       "      <td>313</td>\n",
       "      <td>582</td>\n",
       "      <td>5</td>\n",
       "      <td>69</td>\n",
       "      <td>13770</td>\n",
       "      <td>195716191</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   index  categoryId  channel_subscriberCount  definition  likeCount  \\\n",
       "0      0          13                  1252465           0     6681.0   \n",
       "1      1           5                   328043           0     4001.0   \n",
       "2      2          11                  5003475           0      499.0   \n",
       "3      3           4                   731237           0     2105.0   \n",
       "4      4          14                  5922633           0       66.0   \n",
       "\n",
       "   dislikeCount  viewCount  commentCount  viewCount/channel_month_old  \\\n",
       "0          67.0     235437        1214.0                  3923.950000   \n",
       "1          68.0     221239         541.0                  6913.718750   \n",
       "2          39.0     200147          35.0                  1803.126126   \n",
       "3          53.0     202492         253.0                  1965.941748   \n",
       "4           1.0       4074           2.0                    44.769231   \n",
       "\n",
       "   viewCount/video_month_old  viewCount/http_in_descp  viewCount/NoOfTags  \\\n",
       "0                       6539                    58859               47087   \n",
       "1                      18436                    31605                5822   \n",
       "2                       2599                   100074               18195   \n",
       "3                       6982                   101246               40498   \n",
       "4                        140                     4075                 313   \n",
       "\n",
       "   viewCount/tags_in_desc  social_links  subscriberCount/videoCount  \\\n",
       "0                   78479             4                        3219   \n",
       "1                   13014             7                        1333   \n",
       "2                   13343             4                         227   \n",
       "3                   67497             5                        4202   \n",
       "4                     582             5                          69   \n",
       "\n",
       "   channelViewCount/channeVideoCount  channelViewCount/socialLink  \n",
       "0                             329107                     25604601  \n",
       "1                             120817                      3715148  \n",
       "2                             119078                    524541697  \n",
       "3                             293518                      8512049  \n",
       "4                              13770                    195716191  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.reset_index()\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>level_0</th>\n",
       "      <th>index</th>\n",
       "      <th>categoryId</th>\n",
       "      <th>channel_subscriberCount</th>\n",
       "      <th>definition</th>\n",
       "      <th>likeCount</th>\n",
       "      <th>dislikeCount</th>\n",
       "      <th>viewCount</th>\n",
       "      <th>commentCount</th>\n",
       "      <th>viewCount/channel_month_old</th>\n",
       "      <th>viewCount/video_month_old</th>\n",
       "      <th>viewCount/http_in_descp</th>\n",
       "      <th>viewCount/NoOfTags</th>\n",
       "      <th>viewCount/tags_in_desc</th>\n",
       "      <th>social_links</th>\n",
       "      <th>subscriberCount/videoCount</th>\n",
       "      <th>channelViewCount/channeVideoCount</th>\n",
       "      <th>channelViewCount/socialLink</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>1252465</td>\n",
       "      <td>0</td>\n",
       "      <td>6681.0</td>\n",
       "      <td>67.0</td>\n",
       "      <td>235437</td>\n",
       "      <td>1214.0</td>\n",
       "      <td>3923.950000</td>\n",
       "      <td>6539</td>\n",
       "      <td>58859</td>\n",
       "      <td>47087</td>\n",
       "      <td>78479</td>\n",
       "      <td>4</td>\n",
       "      <td>3219</td>\n",
       "      <td>329107</td>\n",
       "      <td>25604601</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>328043</td>\n",
       "      <td>0</td>\n",
       "      <td>4001.0</td>\n",
       "      <td>68.0</td>\n",
       "      <td>221239</td>\n",
       "      <td>541.0</td>\n",
       "      <td>6913.718750</td>\n",
       "      <td>18436</td>\n",
       "      <td>31605</td>\n",
       "      <td>5822</td>\n",
       "      <td>13014</td>\n",
       "      <td>7</td>\n",
       "      <td>1333</td>\n",
       "      <td>120817</td>\n",
       "      <td>3715148</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>11</td>\n",
       "      <td>5003475</td>\n",
       "      <td>0</td>\n",
       "      <td>499.0</td>\n",
       "      <td>39.0</td>\n",
       "      <td>200147</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1803.126126</td>\n",
       "      <td>2599</td>\n",
       "      <td>100074</td>\n",
       "      <td>18195</td>\n",
       "      <td>13343</td>\n",
       "      <td>4</td>\n",
       "      <td>227</td>\n",
       "      <td>119078</td>\n",
       "      <td>524541697</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>731237</td>\n",
       "      <td>0</td>\n",
       "      <td>2105.0</td>\n",
       "      <td>53.0</td>\n",
       "      <td>202492</td>\n",
       "      <td>253.0</td>\n",
       "      <td>1965.941748</td>\n",
       "      <td>6982</td>\n",
       "      <td>101246</td>\n",
       "      <td>40498</td>\n",
       "      <td>67497</td>\n",
       "      <td>5</td>\n",
       "      <td>4202</td>\n",
       "      <td>293518</td>\n",
       "      <td>8512049</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>4</td>\n",
       "      <td>14</td>\n",
       "      <td>5922633</td>\n",
       "      <td>0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>4074</td>\n",
       "      <td>2.0</td>\n",
       "      <td>44.769231</td>\n",
       "      <td>140</td>\n",
       "      <td>4075</td>\n",
       "      <td>313</td>\n",
       "      <td>582</td>\n",
       "      <td>5</td>\n",
       "      <td>69</td>\n",
       "      <td>13770</td>\n",
       "      <td>195716191</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   level_0  index  categoryId  channel_subscriberCount  definition  likeCount  \\\n",
       "0        0      0          13                  1252465           0     6681.0   \n",
       "1        1      1           5                   328043           0     4001.0   \n",
       "2        2      2          11                  5003475           0      499.0   \n",
       "3        3      3           4                   731237           0     2105.0   \n",
       "4        4      4          14                  5922633           0       66.0   \n",
       "\n",
       "   dislikeCount  viewCount  commentCount  viewCount/channel_month_old  \\\n",
       "0          67.0     235437        1214.0                  3923.950000   \n",
       "1          68.0     221239         541.0                  6913.718750   \n",
       "2          39.0     200147          35.0                  1803.126126   \n",
       "3          53.0     202492         253.0                  1965.941748   \n",
       "4           1.0       4074           2.0                    44.769231   \n",
       "\n",
       "   viewCount/video_month_old  viewCount/http_in_descp  viewCount/NoOfTags  \\\n",
       "0                       6539                    58859               47087   \n",
       "1                      18436                    31605                5822   \n",
       "2                       2599                   100074               18195   \n",
       "3                       6982                   101246               40498   \n",
       "4                        140                     4075                 313   \n",
       "\n",
       "   viewCount/tags_in_desc  social_links  subscriberCount/videoCount  \\\n",
       "0                   78479             4                        3219   \n",
       "1                   13014             7                        1333   \n",
       "2                   13343             4                         227   \n",
       "3                   67497             5                        4202   \n",
       "4                     582             5                          69   \n",
       "\n",
       "   channelViewCount/channeVideoCount  channelViewCount/socialLink  \n",
       "0                             329107                     25604601  \n",
       "1                             120817                      3715148  \n",
       "2                             119078                    524541697  \n",
       "3                             293518                      8512049  \n",
       "4                              13770                    195716191  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = df.reset_index()\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['channel_subscriberCount', 'likeCount', 'channelViewCount/socialLink']"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "continous_name = ['channel_subscriberCount','likeCount','channelViewCount/socialLink']\n",
    "continous_name\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Set Shape (347623, 3)\n",
      "Testing Set Shape (347623,)\n",
      "Training in progress.....\n"
     ]
    }
   ],
   "source": [
    "# Making the training and test set.\n",
    "\n",
    "X = df[continous_name]\n",
    "print (\"Training Set Shape\",X.shape)\n",
    "Y = df.viewCount\n",
    "print(\"Testing Set Shape\",Y.shape)\n",
    "\n",
    "print (\"Training in progress.....\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# parameters\n",
    "n_estimators = 200\n",
    "max_depth = 25\n",
    "min_samples_split=15\n",
    "min_samples_leaf=2\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Random forest classifier\n",
    "clf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=25,\n",
       "           max_features='auto', max_leaf_nodes=None,\n",
       "           min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "           min_samples_leaf=2, min_samples_split=15,\n",
       "           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,\n",
       "           oob_score=False, random_state=None, verbose=0, warm_start=False)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# It is trained of 2 Epochs\n",
    "X = np.concatenate((X,X),axis=0)\n",
    "Y = np.concatenate((Y,Y),axis=0)\n",
    "clf.fit(X,Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature Importance ranking [0.07469696 0.85721645 0.0680866 ]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "print (\"Feature Importance ranking\",clf.feature_importances_)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Saving model to disk\n",
    "pickle.dump(clf, open('model.pkl','wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Number of views in this video will be 1218457 .\n"
     ]
    }
   ],
   "source": [
    "model = pickle.load(open('model.pkl','rb'))\n",
    "e=model.predict([[656024,24885.0,8010063]])\n",
    "e=int(e)\n",
    "print(\"The Number of views in this video will be\",e,\".\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.73258449e+09])"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.predict([[2122121,5552121,55112]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "e=model.predict([[2122121,5552121,55112]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1.73258449e+09])"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "e"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
