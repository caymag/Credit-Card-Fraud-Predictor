import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#PRC used as data has primarily zeros which creates skew and is binary
from sklearn.metrics import precision_recall_curve, auc

# Reads the data utilizing pandas for preprocessing
df = pd.read_csv('creditcard.csv')
print(df.head(5))

# Obtained data and split into x and y variables to help us predict y (Fraud)
x = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Amount', 'Time']]
y = df['Class'].to_numpy()

# Split x and y into testing and training variables, as well as standardizing x
# The training variables are used on the test variables to identify patterns and make predictions
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = .4, random_state = 99)

# Scalar is only on x_train because we want the testing set to not influence the training procees (data leakage)
# What is learned has to come from the models training data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Calculates regression to make prediction
regr = LogisticRegression(max_iter = 100, solver).fit(x_train_scaled, y_train)
prediction = regr.predict(x_test_scaled)
# Outputs info on prediction
print("prediction shape:" , prediction.shape)
print('Accuracy: ',metrics.accuracy_score(y_test, prediction))
print('Recall: ',metrics.recall_score(y_test, prediction, zero_division=1))
print('Precision:',metrics.precision_score(y_test, prediction, zero_division=1))
print('CL Report:',metrics.classification_report(y_test, prediction, zero_division=1))

# Creates PRC graph
precision, recall, thresholds = precision_recall_curve(y_test, prediction)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color = 'blue', lw = 2, label = 'Precision-Recall curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower left")
plt.title('Precision-Recall Curve')
plt.show()

# Confusion mattrix for testing
conf_matrix = metrics.confusion_matrix(y_test, prediction)
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
#plt.show()
