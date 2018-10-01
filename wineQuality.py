import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as pl
import numpy as np
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.metrics import make_scorer
import AuxiliaryFunctions


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) #Project Root
'''
Import the wines dataset
'''
file = ('{}\data_files\winequality.csv'.format(ROOT_DIR))
wines = pd.read_csv(file, sep=';')

'''
Data behavior obseervation
'''

print(wines[0:5])  # verify data structure by looking at the five first rows
print('\nThe dataset contains: ', len(wines), 'specimen.')
print('Missing values:\n', wines.isnull().sum(), '\n\ntotal missing values: ', wines.isnull().sum().sum())
print('\'quality\' fields:\n', wines['quality'].unique(), '\n')

for feat in wines:
    print("FEATURE: {}\n{}\n--- end of {} description ---\n".format(feat, wines[feat].describe(), feat))

'''
Outliers treatment
'''

X = wines.drop('quality', axis=1)
X = X.drop('type', axis=1)
X['alcohol'] = X['alcohol'].str.replace('.', '').astype(np.float64)

y = wines['quality']
tipo = wines['type']

log_data = np.log(X)
for feature in X.keys():
    Q90, Q10 = np.percentile(log_data, [90, 10])
    step = (Q90 - Q10) * 1.5
    minVal = Q10 - step
    maxVal = Q90 + step

    outlier = log_data[~((log_data[feature] >= minVal) & (log_data[feature] <= maxVal))]
    X = X.drop(X.index[outlier.index]).reset_index(drop = True)
    y = y.drop(y.index[outlier.index]).reset_index(drop = True)
    tipo = tipo.drop(tipo.index[outlier.index]).reset_index(drop = True)
    log_data = log_data.drop(log_data.index[outlier.index]).reset_index(drop = True)

'''
Prediction aplication
'''
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),random_state = 22)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
clf = clf.fit(X_train, y_train)
clf = clf.fit(X_train, y_train)
clf_test_predictions = clf.predict(X_test)
clf_train_predictions = clf.predict(X_train)
acc_train_results = accuracy_score(y_train, clf_train_predictions)
acc_test_results = accuracy_score(y_test, clf_test_predictions)

fscore_train_results = fbeta_score(y_train, clf_train_predictions, beta=0.5, average='macro')
fscore_test_results = fbeta_score(y_test, clf_test_predictions, beta=0.5, average='macro')

print("Acurácia teste: {}\t Acurácia treino: {}\nFscore teste: {}\t Fscore treino: {}\n".format(
                acc_test_results, acc_train_results, fscore_test_results, fscore_train_results))

X_predictions = clf.predict(X)

'''
Model Tuning
'''
print("Aplicando tuning do modelo...")
parameters = {'n_estimators':[200,400],
              'learning_rate':[0.1, 0.5, 1.],
              'base_estimator__min_samples_split' : [5,6,7,8],
              'base_estimator__max_depth' :[2,3,4,5]
             }

scorer = make_scorer(fbeta_score,beta=0.5, average='macro')
grid_obj = GridSearchCV(clf, parameters,scorer)
grid_fit = grid_obj.fit(X_train,y_train)
best_clf = grid_fit.best_estimator_
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

print("Best Score: ",grid_fit.best_score_)
print("Best Parameters: ",grid_fit.best_params_)
print ("Unoptimized model\n------")
print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5, average='macro')))
print ("\nOptimized Model\n------")
print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5, average='macro')))

adjusted_predictions = best_clf.predict(X)
csvData = X
csvData['type'] = tipo
csvData['quality'] = y
csvData['predictions'] = adjusted_predictions
csvData.to_csv('adjusted_predictions.csv',sep=';', index=False)

'''
feature importances
'''
importances = clf.feature_importances_
AuxiliaryFunctions.feature_plot(importances, X_train, y_train)

scorer = make_scorer(fbeta_score,beta=0.5)
pl.show()
