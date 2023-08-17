import pandas as pd
from sklearn import metrics

#데이터 import
raw_data = pd.read_csv("C:\\Users\\young\\OneDrive\\바탕 화면\\open\\train.csv")
raw_test_data = pd.read_csv("C:\\Users\\young\\OneDrive\\바탕 화면\\open\\test.csv")

raw_data.head()
raw_test_data.head()

#타입확인
raw_data.dtypes
raw_test_data.dtypes

#null 값 존재하는지 확인 - null 없음
raw_data.isnull().sum()

#class 불균형 확인
raw_data['Outcome'].value_counts()

'''
#이상치 확인
import matplotlib.pyplot as plt

plt.figure(figsize=(30,30))
for col_idx in range(len(raw_data.columns)):
    plt.subplot(6, 2, col_idx + 1)
    plt.boxplot(raw_data[raw_data.columns[col_idx]],flierprops=dict(markerfacecolor='r', marker = 'D'))
    plt.title("Feature (" + raw_data[raw_data.columns[col_idx]] + "):" + raw_data[raw_data.columns[col_idx]], fontsize=20)

plt.show()
'''

#ID 값을 문자열 -> 정수로 변환 -> ID 변수는 필요없을 것 같아 제거
'''
ID = []
for i in range(0, len(raw_data)):
    ID.append(pd.to_numeric(raw_data.loc[i, 'ID'][6:]))

raw_data.insert(1, 'ID_integer', ID)
'''
X = raw_data.loc[:, raw_data.columns.difference(['ID', 'Outcome'])]
y = raw_data['Outcome']

#train_test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10,
                                                stratify=raw_data['Outcome'])


#데이터 스케일링
from sklearn.preprocessing import StandardScaler

standscaler = StandardScaler()
standscaler.fit(X_train)
X_train = standscaler.transform(X_train)
X_test = standscaler.transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

X_train.columns = ['Age', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction', 'Glucose', 'Insulin', 'Pregnancies', 'SkinThickness']
X_test.columns = ['Age', 'BMI', 'BloodPressure', 'DiabetesPedigreeFunction', 'Glucose', 'Insulin', 'Pregnancies', 'SkinThickness']

#RandomForest
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=100)
model = clf.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("RandomForestClassifier 기본 Accurancy: ", metrics.accuracy_score(y_test, y_pred))

#XGBoost
from xgboost import XGBClassifier

clf = XGBClassifier(random_state=100)
model = clf.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("XGBoost 기본 Accurancy: ", metrics.accuracy_score(y_test, y_pred))

#Feature Selection - feature가 6개일 때 제일 성능 좋음
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import KFold

model = RandomForestClassifier()

cv=KFold(n_splits= 3, shuffle=True, random_state=1)

for i in range(1, len(X_train.columns) + 1):
    sfs_xgb = SFS(model,
             k_features = i,
             scoring='accuracy',
             cv = cv)
    sfs1 = sfs_xgb.fit(X_train, y_train)
    sfs1.subsets_
    sfs1.k_feature_idx_
    sfs1.k_feature_names_

#제일 정확도 높은 feature 조합 feature selection
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

FS1_rf_df = X_train[['Age', 'BMI', 'BloodPressure', 'Glucose', 'Insulin', 'Pregnancies']]
FS1_test_rf_df = X_test[['Age', 'BMI', 'BloodPressure', 'Glucose', 'Insulin', 'Pregnancies']]
FS1_rf_model = RandomForestClassifier()

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = [6]
max_depth = [int(x) for x in np.linspace(5, 20, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

pram_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split' : min_samples_split,
                'min_samples_leaf' : min_samples_leaf,
                'bootstrap': bootstrap}

FS1_ramdomForest = RandomizedSearchCV(estimator= FS1_rf_model, param_distributions=pram_grid,
                           cv = cv, n_jobs=-1, verbose = 2, random_state=50)
FS1_ramdomForest.fit(FS1_rf_df, y_train)
FS1_ramdomForest.best_params_ 
best_rf1_model = FS1_ramdomForest.best_estimator_
y_fsrf_pred = best_rf1_model.predict(FS1_test_rf_df)
print("RandomForest - FS & Parameter tuning 이후 Accurancy: ", metrics.accuracy_score(y_test, y_fsrf_pred))

#Permutation_importance - feature가 6개일 때 가장 성능 좋음
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import numpy as np

r_model = RandomForestClassifier(n_estimators=10)
r_model.fit(X_train, y_train)

r_result = permutation_importance(r_model, X_train, y_train, n_repeats= 5, random_state= 40, n_jobs = -1 )
sorted_rf_importances_idx = r_result.importances_mean.argsort()
r_importances = pd.DataFrame(
    r_result.importances_mean[sorted_rf_importances_idx].T,
    index = X_train.columns[sorted_rf_importances_idx],
)
for i in range(1, len(X_train.columns) + 1):
    rf_index = r_importances[len(r_importances) - i : len(r_importances) + 1].index
    df_rf = X_train[rf_index]
    r_select_model = RandomForestClassifier(n_estimators=10)
    r_scores = cross_val_score(r_select_model, df_rf, y_train, cv=5)
    print("{} 개의 {}를 사용했을 때 정확도 : {}" .format(i, rf_index, np.mean(r_scores)))

FS2_rf_df = X_train[['Insulin', 'DiabetesPedigreeFunction', 'Pregnancies', 'Age', 'BMI','Glucose']]
FS2_test_rf_df = X_test[['Insulin', 'DiabetesPedigreeFunction', 'Pregnancies', 'Age', 'BMI','Glucose']]
FS2_rf_model = RandomForestClassifier()

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = [6]
max_depth = [int(x) for x in np.linspace(5, 20, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

pram_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split' : min_samples_split,
                'min_samples_leaf' : min_samples_leaf,
                'bootstrap': bootstrap}

FS2_RandomForest = RandomizedSearchCV(estimator= FS2_rf_model, param_distributions=pram_grid,
                           cv = cv, n_jobs=-1, verbose = 2)
FS2_RandomForest.fit(FS2_rf_df, y_train)

FS2_RandomForest.best_params_
best_rf_model = FS2_RandomForest.best_estimator_
y_fsrf2_pred = best_rf_model.predict(FS2_test_rf_df)
print("RandomForest - FS(permutation importance) & Parameter tuning 이후 Accurancy: ", 
                                metrics.accuracy_score(y_test, y_fsrf2_pred))


#Random Forest hyper-parameter tuning
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = [5, 6, 8, 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 20, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

pram_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split' : min_samples_split,
                'min_samples_leaf' : min_samples_leaf,
                'bootstrap': bootstrap}

rf = RandomForestClassifier(random_state = 1)
FS3_RandomForest = RandomizedSearchCV(estimator= rf, param_distributions=pram_grid,
                           cv = cv, n_jobs=-1, verbose = 2)
FS3_RandomForest.fit(X_train, y_train)

FS3_RandomForest.best_params_
best_rf_model = FS3_RandomForest.best_estimator_
y_fsrf3 = best_rf_model.predict(X_test)
print("RandomForest - Parameter tuning 이후 Accurancy: ", metrics.accuracy_score(y_test, y_fsrf3))

#XGBoost Feature Selection(SFS) - 
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import KFold

cv=KFold(n_splits= 3, shuffle=True, random_state=1)

for i in range(1, len(X_train.columns) + 1):
    sfs_xgb = SFS(model,
             k_features = i,
             scoring='accuracy',
             cv = cv)
    sfs1 = sfs_xgb.fit(X_train, y_train)
    sfs1.subsets_
    sfs1.k_feature_idx_
    sfs1.k_feature_names_


from sklearn.model_selection import RandomizedSearchCV

FS1_xg_df = X_train[['Age', 'BMI', 'DiabetesPedigreeFunction', 'Glucose']]
FS1_test_xg_df = X_test[['Age', 'BMI', 'DiabetesPedigreeFunction', 'Glucose']]
FS1_xg_model = XGBClassifier()

param_grid={'booster' :['gbtree'],
            'n_estimators':[50,100],
            'max_depth':[5,6,8],
            'min_child_weight':[1,3,5],
            'gamma':[0,1,2,3],
            'objective':['binary:logistic'],
            'random_state':[1]}

cv = KFold(n_splits=5, shuffle=True, random_state=1)
model = XGBClassifier()

FS1_XGBoost = RandomizedSearchCV(model, param_distributions=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
FS1_XGBoost.fit(FS1_xg_df, y_train)
FS1_XGBoost.best_params_
best_rf_model = FS1_XGBoost.best_estimator_
y_xgfs1_pred = FS1_XGBoost.predict(FS1_test_xg_df)
print("XGBoost Accurancy(SFS) : ", metrics.accuracy_score(y_test, y_xgfs1_pred))


#XGBoost Feature Selection(Permutation)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import numpy as np

x_model = XGBClassifier()
x_model.fit(X_train, y_train)

x_result = permutation_importance(x_model, X_train, y_train, n_repeats= 5, random_state= 40, n_jobs = -1)

sorted_xgb_importances_idx = x_result.importances_mean.argsort()
x_importances = pd.DataFrame(
    x_result.importances_mean[sorted_xgb_importances_idx].T,
    index = X_train.columns[sorted_xgb_importances_idx],
)
x_importances.columns = ['importances']
print(x_importances)

for i in range(1, len(X_train.columns)+1):
    xgb_index = x_importances[len(x_importances) - i : len(x_importances) + 1].index
    df_xgb = X_train[xgb_index]
    xgb_select_model = XGBClassifier()
    xgb_scores = cross_val_score(xgb_select_model, df_xgb, y_train, cv=5)
    print("{} 개의 {}를 사용했을 때 정확도 : {}" .format(i, xgb_index, np.mean(xgb_scores)))

FS2_xg_df = X_train[['DiabetesPedigreeFunction', 'Age', 'BMI', 'Glucose']]
FS2_test_xg_df = X_test[['DiabetesPedigreeFunction', 'Age', 'BMI', 'Glucose']]
FS2_xg_model = XGBClassifier()

pram_grid={'booster' :['gbtree'],
            'n_estimators':[50,100],
            'max_depth':[5,6,8],
            'min_child_weight':[1,3,5],
            'gamma':[0,1,2,3],
            'objective':['binary:logistic'],
            'random_state':[1]}

cv = KFold(n_splits=5, shuffle=True, random_state=1)

FS2_XGBoost = RandomizedSearchCV(estimator= FS2_xg_model, param_distributions=pram_grid,
                           cv = cv, n_jobs=-1, verbose = 2)
FS2_XGBoost.fit(FS2_xg_df, y_train)

FS2_XGBoost.best_params_
best_rf_model = FS2_XGBoost.best_estimator_
y_fsxg2_pred = best_rf_model.predict(FS2_test_xg_df)
print("XGBoost - Permutation_importance & Parameter tuning 이후 Accurancy: ", 
                                metrics.accuracy_score(y_test, y_fsxg2_pred))


#XGBoost hyper-parameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

param_grid={'booster' :['gbtree'],
            'n_estimators':[50,100, 150, 200],
            'max_depth':[5,6,8],
            'min_child_weight':[1,3,5],
            'gamma':[0,1,2,3],
            'objective':['binary:logistic']}

cv=KFold(n_splits=5, shuffle=True, random_state=1)

model = XGBClassifier()
gcv=RandomizedSearchCV(model, param_distributions=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

gcv.fit(X_train, y_train)
gcv.best_params_
best_rf_model = gcv.best_estimator_
y_pred = gcv.predict(X_test)
print("XGBoost - Parameter tuning 이후 Accurancy: ", 
                                metrics.accuracy_score(y_test, y_pred))

#Model Stacking - RandomForest와 Xgboost 선택
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', FS2_RandomForest.best_estimator_),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(C=1.0, max_iter = 120000, random_state=30, dual = False))),
    ('lr', LogisticRegression(solver='lbfgs', max_iter=120000, random_state=30))]

model_1 = StackingClassifier(estimators=estimators, final_estimator=gcv.best_estimator_)
scores_1 = cross_val_score(model_1, X_train, y_train, cv=5)
np.mean(scores_1)

#전체 데이터 학습
X = raw_data.loc[:, raw_data.columns.difference(['ID', 'Outcome'])]
y = raw_data['Outcome']
X_test = raw_test_data.loc[:, raw_data.columns.difference(['ID', 'Outcome'])]

result_model = model_1.fit(X, y)
y_pred = result_model.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns=['Outcome'])
merge__pred = pd.concat([raw_test_data['ID'], y_pred], axis = 1)
merge__pred.to_csv("C:\\Users\\young\\OneDrive\\바탕 화면\\대학\\학부연구생\\dacon_당뇨병진단\\Result.csv", index=False)