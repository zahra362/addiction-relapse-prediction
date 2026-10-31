import pandas as pd
import numpy as np

from MyFunctions import *

df = pd.read_excel('dataset.xlsx')

df.loc[:, 'تاریخ شروع درمان'] = df['تاریخ شروع درمان'].apply(lambda x: convert_date(x))
df.loc[:, 'آخرین تاریخ مراجعه'] = df['آخرین تاریخ مراجعه'].apply(lambda x: convert_date(x))
df.loc[:, 'تاریخ شروع درمان'] = pd.to_datetime(df['تاریخ شروع درمان'])
df.loc[:, 'آخرین تاریخ مراجعه'] = pd.to_datetime(df['آخرین تاریخ مراجعه'])
df.loc[:, 'طول درمان'] = (df['آخرین تاریخ مراجعه'] - df['تاریخ شروع درمان']).dt.days

df.drop(columns=['آخرین تاریخ مراجعه', 'تاریخ شروع درمان', 'ID'], inplace=True)

tmp_True, my_estimators_True, my_features_True, end_clf_True, x_test_True, y_test, x_train_new_True, y_train_True = test_model(
    df, isImpute=True, isSMOTE=True, isNormalization=True, k=35, n_estimator=100, max_rows=100, max_features=7)
tmp_False, my_estimators_False, my_features_False, end_clf_False, x_test_False, y_test, x_train_new_False, y_train_False = test_model(
    df, isImpute=True, isSMOTE=False, isNormalization=True, k=35, n_estimator=50, max_rows=100, max_features=7)

n_estimator_False = 50
n_estimator_True = 100

x_test_new = np.empty(shape=(x_test_False.shape[0], n_estimator_False))

for i, clf in enumerate(my_estimators_False):
    x_test_new[:, i] = clf.predict(x_test_False[:, my_features_False[i]])

y_hat_False = end_clf_False.predict_proba(x_test_new)
y_hat_False_b = end_clf_False.predict(x_test_new)

x_test_new = np.empty(shape=(x_test_True.shape[0], n_estimator_True))

for i, clf in enumerate(my_estimators_True):
    x_test_new[:, i] = clf.predict(x_test_True[:, my_features_True[i]])

y_hat_True = end_clf_True.predict_proba(x_test_new)
y_hat_True_b = end_clf_True.predict(x_test_new)

y_hat_all = np.concatenate((y_hat_True, y_hat_False), axis=1)
y_hat_over = np.apply_along_axis(predict_class, arr=y_hat_all, axis=1)

output_dict = {"Recall Score": recall_score(y_test, y_hat_over),
               "Accuracy Score": accuracy_score(y_test, y_hat_over),
               "Precision Score": precision_score(y_test, y_hat_over),
               "F1 Score": f1_score(y_test, y_hat_over),
               "Confusion Matrix": confusion_matrix(y_test, y_hat_over)}

print(output_dict)
