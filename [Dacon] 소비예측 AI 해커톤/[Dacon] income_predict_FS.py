import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from random import randint

warnings.filterwarnings("ignore", category=FutureWarning)

# VIF를 계속 갱신하면서 계산
def vif_method(x) :
    # vif가 10 초과 시 drop하기 위한 임계값
    thresh = 10
    output = pd.DataFrame()
    #데이터 칼럼 개수
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    
    for i in range(1, k) :
        print(f'{i}번째 VIF 측정')
        #VIF 최대 값 선정
        a = np.argmax(vif)
        if(i == 1) :
            print(f'Max VIF feature & value : {x.columns[a], {vif[a]}}')
        else :
            print(f'Max VIF feature & value : {output.columns[a], {vif[a]}}')
        
        if(vif[a] <= thresh):
            print('\n')
            for q in range(output.shape[1]) :
                print(f'{output.columns[q]}의 vif는 {np.round(vif[q], 2)}입니다.')
            break
        if(i == 1) :
            output = x.drop(x.columns[a], axis=1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        else :
            output = output.drop(output.columns[a], axis=1)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]     
                   
    return output

df = pd.read_csv("C:\\Users\\young\\Downloads\\open (1)\\open\\train.csv")
df_test = pd.read_csv("C:\\Users\\young\\Downloads\\open (1)\\open\\test.csv")

df.shape

df.describe()

#X와 y 분리하기
df_X = df.drop(columns=['ID', 'Income'])
df_y = df['Income']
df_test = df_test.drop(columns='ID')


###########################################################################
# Encoding 및 표준화
###########################################################################

#수치형 데이터는 표준화
scaler = StandardScaler()

numerical_cols = df_X.select_dtypes(include = ['int', 'float']).columns

df_X[numerical_cols] = scaler.fit_transform(df_X[numerical_cols])
df_test[numerical_cols] = scaler.fit_transform(df_test[numerical_cols])

#데이터 인코딩 - Label encoding으로만 진행
label_encoder = LabelEncoder()

object_cols = df_X.select_dtypes('object').columns

for col in object_cols :
    df_X[col] = label_encoder.fit_transform(df_X[col])
    df_test[col] = label_encoder.fit_transform(df_test[col])


#아무런 처리 안했을 때 RMSE
x_tr, x_te, y_tr, y_te = train_test_split(df_X, df_y, random_state=0)
model = RandomForestRegressor(random_state=1234)
model = model.fit(x_tr, y_tr)
y_pr = model.predict(x_te)

RMSE = mean_squared_error(y_te, y_pr)**0.5
print(RMSE)
######################################################################################
# Feature Selection - filter method
######################################################################################

#Peason's Correlation
data = pd.DataFrame(df_X, columns = df_X.columns)
data['Income'] = df_y

#Income을 target으로 상관관계 분석
target_correlation = data.corr()[['Income']]
plt.figure(figsize=(7, 5))
sns.heatmap(target_correlation, annot=True, cmap = plt.cm.Reds)  #annot은 각 셀에 값을 주석으로 추가한 것
plt.show()

# 변수끼리의 상관도 확인
sns.heatmap(data.corr(), annot=True, cmap = plt.cm.Reds)
plt.show()

# VIF 도출
data_X = data.drop(columns='Income')
vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(data_X.values, i) for i in range(data_X.shape[1])]
vif['features'] = df_X.columns

## VIF 값이 높은 순으로 정렬
vif = vif.sort_values(by='VIF Factor', ascending=False)

## VIF가 높은 상위 Feature들을 for문을 통해 제거해가면서 확인
vif_X = vif_method(data_X)

df_vif = df_X.drop(columns = ['Household_Summary', 'Birth_Country (Father)', 'Birth_Country (Mother)'])
df_test = df_test.drop(columns = ['Household_Summary', 'Birth_Country (Father)', 'Birth_Country (Mother)'])

X_train, X_test, y_train, y_test = train_test_split(df_vif, df_y, random_state=0)
model = RandomForestRegressor(random_state=1234)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

RMSE = mean_squared_error(y_test, y_pred)**0.5
print(RMSE)

######################################################################################
# 하이퍼파라미터 공간 정의
param_dist = {
    'n_estimators': (100,200,300,400,500),
    'max_depth': [None] + list(np.arange(3, 20, 2)),
    'min_samples_split': list(range(2, 21, 2)),
    'min_samples_leaf': list(range(1, 21, 2)),
    'max_features': ['sqrt', 'log2', None]
}

# 랜덤 서치를 사용한 하이퍼파라미터 튜닝
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

test_predict = random_search.predict(df_test)

test = pd.DataFrame()
df_test = pd.read_csv("C:\\Users\\young\\Downloads\\open (1)\\open\\test.csv")
test['Income'] = test_predict
test['ID'] = df_test['ID']
test = test[['ID', 'Income']]
test.to_csv("C:\\Users\\young\\Downloads\\open (1)\\open\\answer.csv", index=False)

