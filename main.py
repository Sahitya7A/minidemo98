#Importing Required Libraries
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt

#Reading the Data Set(.tsv file) using Pandas
path = 'https://raw.githubusercontent.com/Sahitya7A/minidemo98/main/data23.tsv'
data = pd.read_table(path,header=None,skiprows=1,names=['Sentiment','Review'])
X = data.Review
y = data.Sentiment
#Converting text into tokens/features using CountVectorizer
vect = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size= 0.2)
#Transforming text(train data) into counts of features(frequency)
#Tokenization and Vectorization
vect.fit(X_train)
#Data Encoding-Transformation
X_train_dtm = vect.transform(X_train) 
X_test_dtm = vect.transform(X_test)

#Accuracy using Naive Bayes Model
NB = MultinomialNB()
NB.fit(X_train_dtm, y_train)
y_pred = NB.predict(X_test_dtm)
print('\nNaive Bayes')
print('Accuracy Score: ',metrics.accuracy_score(y_test,y_pred)*100,'%',sep='')
print('F1 Score: ',metrics.f1_score(y_test,y_pred))
print('Confusion Matrix: ',metrics.confusion_matrix(y_test,y_pred), sep = '\n')

'''
Naive Bayes
Accuracy Score: 80.58510638297872%
F1 Score:  0.8188585607940446
Confusion Matrix: 
[[138  38]
 [ 35 165]]
 '''

#Naive Bayes Analysis
token_words = vect.get_feature_names()
print('\nAnalysis')
print('No. of tokens: ',len(token_words))
counts = NB.feature_count_
df_table = {'Token':token_words,'Negative': counts[0,:],'Positive': counts[1,:]}
tokens = pd.DataFrame(df_table, columns= ['Token','Positive','Negative'])
positives = len(tokens[tokens['Positive']>tokens['Negative']])
print('No. of positive tokens: ',positives)
print('No. of negative tokens: ',len(token_words)-positives)

'''

Analysis
No. of tokens:  2297
No. of positive tokens:  846
No. of negative tokens:  1451
/usr/local/lib/python3.7/dist-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
  warnings.warn(msg, category=FutureWarning)
  '''

#Checking positivity/negativity of specific tokens
token_search = ['awesome']
print('\nSearch Results for token/s:',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])
token_search = ['disapointed']
print('\nSearch Results for token/s:',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])
token_search = ['happy']
print('\nSearch Results for token/s:',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])
token_search = ['worse']
print('\nSearch Results for token/s:',token_search)
print(tokens.loc[tokens['Token'].isin(token_search)])

'''

Search Results for token/s: ['awesome']
       Token  Positive  Negative
147  awesome      34.0       2.0

Search Results for token/s: ['disapointed']
           Token  Positive  Negative
549  disapointed       0.0       5.0

Search Results for token/s: ['happy']
     Token  Positive  Negative
918  happy      21.0      11.0

Search Results for token/s: ['worse']
      Token  Positive  Negative
2266  worse       0.0      34.0
'''

#Custom Test: Test a review on the performing model
trainingVector = CountVectorizer(stop_words='english', ngram_range = (1,1), max_df = .80, min_df = 5)
trainingVector.fit(X)
X_dtm = trainingVector.transform(X)
Naive = MultinomialNB()
Naive.fit(X_dtm, y)
#Input Review
print('Test a customer review message')
print('Enter a customer review to be analysed: ', end=" ")
test = []
test.append(input())
test_dtm = trainingVector.transform(test)
predLabel = Naive.predict(test_dtm)
tags = ['Negative','Positive']
#Displaying Output
print('The review is predicted as',tags[predLabel[0]])

'''
Test a customer review message
Enter a customer review to be analysed:  it was worst
The review is predicted as Negative
'''

total=len(data.Sentiment)+1#total no of reviews
pos=np.count_nonzero(data['Sentiment'])#positive reviews
neg=total-pos#negative reviews
#print(pos,neg)
slices=[pos,neg]
dept=['positive','negative']
cols=['green','red']
plt.pie(slices,labels=dept,colors=cols,autopct='%.1f%%')
plt.title('Percentage of product\'s status in market.')
plt.show()

slices=[1451,846]
dept=['positive','negative']
cols=['green','red']
plt.pie(slices,labels=dept,colors=cols,autopct='%.1f%%')
plt.title('Percentage of product\'s status in market.')
plt.show()

slices=[846,1451]
dept=['positive','negative']
cols=['green','red']
plt.pie(slices,labels=dept,colors=cols,autopct='%.1f%%')
plt.title('Percentage of product\'s status in market.')
plt.show()




