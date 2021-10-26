import numpy as np
import pandas as pd
import seaborn as sns
import neattext.functions as nfx

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from unidecode import unidecode
from sklearn.preprocessing import LabelEncoder

train_df=pd.read_csv('/content/drive/MyDrive/abc.csv')
print(train_df['sentiment'].value_counts())
sns.countplot(x='sentiment',data=train_df)
train_df.head(10)
data= train_df.loc[:,['content','sentiment']]

train_df['clean_text'] = train_df['content'].apply(nfx.remove_userhandles)
train_df['clean_text'] = train_df['content'].apply(nfx.remove_stopwords)
x=train_df['clean_text']
target=data['sentiment']
y=target

x_train, x_test, y_train, y_test = train_test_split(x.values,y.values,test_size=0.3,random_state=42)

text_clf=Pipeline([('vect',CountVectorizer()),('tfidf',TfidfTransformer()),('lr',LogisticRegression())])
text_clf.fit(x_train,y_train)


import pickle
import gzip
filename="sp18-bcs-082.pkl"
pickle.dump(text_clf,gzip.open(filename,'wb'))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
predictions = text_clf.predict(x_test)
print(confusion_matrix(y_test,predictions))
print('Accuracy Score',accuracy_score(y_test,predictions)*100,'%')

#print('Accuracy Score',text_clf.score(x_test,y_test))



ex1 = "I do not sleep"
# ex1 = "i feel dispapoint"
print(text_clf.predict([ex1]))
print(text_clf.predict_proba([ex1]))
print(text_clf.classes_)





from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
prec = metrics.precision_score(y_test, predictions,average='micro')
print("Precision for multiclassnavie base :",prec)
recall = metrics.recall_score(y_test, predictions,average='micro')
print("Recall for Multiclass Navie base:",recall)
Fmeasure = (2*prec*recall)/(prec+recall)
print("The F-Measure Multiclass Navie base is: ",Fmeasure)

cnf_matrix = metrics.confusion_matrix(y_test,predictions)

fp_nb = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
fn_nb = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
tp_nb = np.diag(cnf_matrix)
tn_nb = cnf_matrix.sum() - (fp_nb + fp_nb + fp_nb)

fp_nb = fp_nb.astype(float)
fn_nb = fn_nb.astype(float)
tp_nb = tp_nb.astype(float)
tn_nb = tn_nb.astype(float)

TPR_nb = tp_nb/(tp_nb+fn_nb)
tpr_nb =0
for i in TPR_nb:
  tpr_nb +=i
tpr_nb = tpr_nb/len(TPR_nb)   
print("True Positive rate for naive bayes is: ", tpr_nb)  


FPR_nb = fp_nb/(fp_nb+tn_nb)
fpr_nb =0
for i in FPR_nb:
  fpr_nb +=i
fpr_nb = fpr_nb/len(FPR_nb)
print("False Positive rate for naive bayes is: ", fpr_nb)  

from matplotlib import pyplot as plt
plt.figure(figsize=(5,5),dpi=100)
plt.plot(FPR_nb,TPR_nb,marker='.',label='ROC')
plt.xlabel('False positive Rate')
plt.ylabel('True positive Rate')
plt.legend()
plt.show()

sns.heatmap(confusion_matrix(y_test,predictions), annot=True, cmap='Blues')