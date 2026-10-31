import jdatetime
import numpy as np
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import jdatetime
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.decomposition import PCA, TruncatedSVD
from copy import copy
from sklearn.model_selection import GridSearchCV
import tqdm


def split_train_data(x_train, y_train, max_rows=200):
    x_index_false = np.argwhere(y_train == 0)
    x_index_true = np.argwhere(y_train == 1)
    #     x_false_size = int(max_rows*(x_index_false.shape[0]/(x_index_false.shape[0] + x_index_true.shape[0])))
    x_true_size = int(max_rows / 2)
    tmp_1 = np.random.permutation(x_index_true)[:x_true_size]
    tmp_2 = np.random.permutation(x_index_false)[:(max_rows - tmp_1.shape[0])]
    return np.concatenate((tmp_1, tmp_2))


def my_error(y_true, y_pred):
    my_score_dict = {'recall_score': recall_score,
                     'accuracy_score': accuracy_score,
                     'precision_score': precision_score,
                     'f1_score': f1_score}
    fitness_sum = 0
    for i, j in my_score_dict.items():
        tmp = j(y_true, y_pred)
        if tmp > 0.5:
            fitness_sum += 0.5
        else:
            fitness_sum -= 0.5
    return fitness_sum


def convert_date(x):
    years = x // 10000 + 1300
    months = (x % 10000) // 100
    days = x % 100
    return jdatetime.date(years, months, days).togregorian()


def test_model(df,
               isImpute=False,
               isFeatureSelection=True,
               isSMOTE=False,
               isNearMiss=False,
               isPCA=False,
               isNormalization=True,
               k=30,
               n_estimator=100,
               max_rows=100,
               max_features=3):
    np.random.seed(42)
    features = df.drop(columns=['رویگردانی']).columns
    target = 'رویگردانی'

    X = df[features].values
    y = df[target].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    if isImpute:
        imp = IterativeImputer()

        x_train = imp.fit_transform(x_train)
        x_test = imp.transform(x_test)
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        x_train = imp.fit_transform(x_train)
        x_test = imp.transform(x_test)

    if isNormalization:
        my_norm = StandardScaler()
        x_train = my_norm.fit_transform(x_train)
        x_test = my_norm.transform(x_test)

    if isFeatureSelection:
        #         k = 30
        my_fs = SelectKBest(f_classif, k=k)
        x_train = my_fs.fit_transform(x_train, y_train)
        x_test = my_fs.transform(x_test)

    if isSMOTE:
        my_smp = SMOTE(n_jobs=-1)
        x_train, y_train = my_smp.fit_resample(x_train, y_train)

    if isNearMiss:
        my_smp = NearMiss(n_jobs=-1)
        x_train, y_train = my_smp.fit_resample(x_train, y_train)

    my_estimators = []
    my_features = []
    my_rows = []
    for i in range(n_estimator):
        tmp_features = np.random.permutation(np.arange(k))[:max_features]
        #     tmp_rows = np.random.permutation(np.arange(x_train.shape[0]))[:max_rows]
        tmp_rows = split_train_data(x_train, y_train, max_rows=max_rows)
        clf = DecisionTreeClassifier(criterion='entropy', class_weight={0: 0.1, 1: 1})
        clf.fit(x_train[tmp_rows.reshape(-1, 1), tmp_features], y_train[tmp_rows])
        my_estimators.append(copy(clf))
        my_features.append(copy(tmp_features))
        my_rows.append(copy(tmp_rows.reshape(-1, 1)))

    x_train_new = np.empty(shape=(x_train.shape[0], n_estimator))

    for i, clf in enumerate(my_estimators):
        x_train_new[:, i] = clf.predict(x_train[:, my_features[i]])

    #     my_param = {'alpha':[0.2, 0.5, 0.7, 1, 1.1, 1.2, 1.3, 1.5]}
    #     my_GSCV = GridSearchCV(MultinomialNB(), my_param, scoring=make_scorer(my_error, greater_is_better=True))
    #     my_GSCV = GridSearchCV(MultinomialNB(), my_param, scoring=make_scorer(f1_score, greater_is_better=True))
    #     my_GSCV.fit(x_train_new, y_train)
    #     end_clf = my_GSCV.best_estimator_

    end_clf = MultinomialNB()
    end_clf.fit(x_train_new, y_train)

    #     print(f"Train Score is: {end_clf.score(x_train_new, y_train)}")

    x_test_new = np.empty(shape=(x_test.shape[0], n_estimator))

    for i, clf in enumerate(my_estimators):
        x_test_new[:, i] = clf.predict(x_test[:, my_features[i]])

    y_hat = end_clf.predict(x_test_new)
    output_dict = {"Recall Score": recall_score(y_test, y_hat),
                   "Accuracy Score": accuracy_score(y_test, y_hat),
                   "Precision Score": precision_score(y_test, y_hat),
                   "F1 Score": f1_score(y_test, y_hat),
                   "Confusion Matrix": confusion_matrix(y_test, y_hat)}
    return output_dict, my_estimators, my_features, end_clf, x_test, y_test, x_train_new, y_train


def predict_class(y_hat):
    if y_hat[2] <= 0.3 and y_hat[1] <= 0.8:
        return True if y_hat[3] <= 0.8 else False
    elif y_hat[2] <= 0.3 and y_hat[1] > 0.8:
        if y_hat[2] <= 0.007:
            return False
        else:
            return True if y_hat[1] <= 0.998 else False
    else:
        if y_hat[0] <= 0.2 and y_hat[1] <= 0.91:
            return False if y_hat[1] <= 0.8 else True
        elif y_hat[0] <= 0.2 and y_hat[1] > 0.91:
            return False
        elif y_hat[1] <= 0.56:
            return False
        else:
            return False if y_hat[0] <= 0.38 else True
