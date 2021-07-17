#%%
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# %%
raw_train_data = pd.read_csv('./data/train.csv')
raw_test_data = pd.read_csv('./data/test.csv')
# %%
raw_train_data.head()
#2016-02-01~2021-01-26
# %%
raw_train_data.tail()
#%%
raw_train_data.info()
#%%
raw_test_data.head()
#2021-01-27~2021-04-09
#%%
raw_test_data.tail()
# %%
raw_test_data.info()
# %%
## 데이터 정제
# 데이터 복사
train_data = raw_train_data.copy()
test_data = raw_test_data.copy()
# %%
# feature name을 영어에서 한글로 변경
train_data.columns = ['date', 'day', 'employees','day_off', 'business_trip', 'overtime', 'telecommuting', 'breakfast', 'lunch', 'dinner', 'num_lunch', 'num_dinner']
test_data.columns = ['date', 'day', 'employees','day_off', 'business_trip', 'overtime', 'telecommuting', 'breakfast', 'lunch', 'dinner']
# %%
# feature 'day'의 한글 데이터를 영어로 변경
day_dict = {'월': 'Mon', '화':'Tue', '수':'Wed', '목':'Thu', '금':'Fri'}
train_data['day'] = train_data['day'].map(day_dict)
test_data['day'] = test_data['day'].map(day_dict)
# %%
train_data.head()

# %%
test_data.head()
# %%
# 요일 별 점심 식수 boxplot
fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams['font.size'] = 30
ax = sns.boxplot(x='day', y='num_lunch', data=train_data)
ax = sns.swarmplot(x='day', y='num_lunch', data=train_data, color=".25")
plt.show()
# 월요일 - 출근하는 사람이 많다. 
# 금요일 - 출근하는 사람이 적다.
# %%
# 요일 별 저녁 식수 boxplot
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='day', y='num_dinner', data=train_data)
ax = sns.swarmplot(x='day', y='num_dinner', data=train_data, color=".25")
plt.show()
# 월요일 - 야근하는 확률이 높다.
# 수요일 - 야근하는 확률이 확실하게 낮다. 자기계발의날 
# 금요일 - 야근하는 확률이 낮다.
# %%
# 요일 별 day_off boxplot
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='day', y='day_off', data=train_data)
ax = sns.swarmplot(x='day', y='day_off', data=train_data, color=".25")
plt.show()
# 금요일에 휴가가는 확률이 약간 높다.
# %%
# 요일 별 business_trip boxplot
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='day', y='business_trip', data=train_data)
ax = sns.swarmplot(x='day', y='business_trip', data=train_data, color=".25")
plt.show()
# 월요일 -> 금요일로 갈수록 출장가는 확률이 높다.
# %%
# 요일 별 overtime boxplot
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='day', y='overtime', data=train_data)
ax = sns.swarmplot(x='day', y='overtime', data=train_data, color=".25")
plt.show()
# 월요일, 화요일 - 야근하는 확률이 높다. 
# 수요일 - 야근을 아예 하지 않는다.
# 금요일 - 야근하는 확률이 낮다. 
# %%
# 요일 별 telecommuting boxplot
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='day', y='telecommuting', data=train_data)
ax = sns.swarmplot(x='day', y='telecommuting', data=train_data, color=".25")
plt.show()
# 재택근무와 요일은 무관하다.
# %%
start_tele = train_data.loc[train_data['telecommuting']>0, 'date'].min()
print('Start of telecommuting : {}'.format(start_tele))
# 재택근무는 2020-01-06부터 시작했다. 
# %%
# feature 간의 관계 파악
fig, ax = plt.subplots(figsize = (20, 10), ncols = 4, nrows = 2, sharey=True)
plt.rcParams['font.size'] = 30
sns.color_palette("Paired")
features = ['employees','day_off', 'business_trip', 'overtime', 'employees','day_off', 'business_trip', 'overtime']

for i, feature in enumerate(features):
    row = i // 4
    col = i % 4
    if i < 4:
        sns.regplot(x=feature, y = 'num_lunch', data = train_data, ax = ax[row][col], color = 'salmon', marker = '+')
    else: 
        sns.regplot(x=feature, y = 'num_dinner', data = train_data, ax = ax[row][col],  color = 'skyblue', marker = '+')
# %%
# feature correlation heatmap
corr_data = train_data[['num_lunch', 'num_dinner', 'employees','day_off', 'business_trip', 'overtime', 'telecommuting']]
mask = np.triu(np.ones_like(corr_data.corr(), dtype=np.bool))
plt.rcParams['font.size'] = 30

fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(corr_data.corr(), 
            annot=True, 
            cmap="YlGnBu", 
            mask = mask)
ax.set_title('Correlation Heatmap', pad = 10)
plt.show()
# %%
# 재택근무 유무 기준으로 covid-19 발생 전후 분간
train_data['covid-19'] = 0
# %%
# 2020-01-06부터 재택근무하는 사람이 존재하므로 covid-19 발생시작날짜로 지정
train_data.loc[train_data[train_data['date'] == '2020-01-06'].index[0]:]['covid-19'] = 1
# %%
train_data.loc[train_data[train_data['date'] == '2020-01-06'].index[0]]
# %%
# 재택근무 전후 요일 별 점심 식수 boxplot 비교
fig, ax = plt.subplots(figsize=(10, 10))
plt.rcParams['font.size'] = 30
ax = sns.boxplot(x='day', y='num_lunch', hue='covid-19', data=train_data)
plt.show()
# 재택근무 시행 후 약간 요일 별 식수 평균이 감소했으나 큰 차이는 없다.
# %%
# 재택근무 전후 요일 별 저녁 식수 boxplot 비교
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='day', y='num_dinner', hue='covid-19', data=train_data)
plt.show()
# 재택근무 시행 후 약간 요일 별 식수 평균
# 월요일은 큰 차이 없으나 나머지 요일은 확실히 감소했다.
# %%
# 재택근무 전후 요일 별 day_off boxplot 비교
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='day', y='day_off', hue='covid-19', data=train_data)
plt.show()
# 휴가가는 인원의 평균이 약간 증가했다. 
# %%
# 재택근무 전후 요일 별 business_trip boxplot 비교
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='day', y='business_trip', hue='covid-19', data=train_data)
plt.show()
# 코로나의 여파로 출장가는 비율이 감소했다.
# %%
# 재택근무 전후 요일 별 overtime boxplot 비교
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(x='day', y='overtime', hue='covid-19', data=train_data)
plt.show()
# 오히려 재택근무 시행 후 야근하는 횟수가 증가했다. 
# %%
# 근로자 수 추이 파악
train_data.plot(x = 'date', y = 'employees', figsize = (40, 8))
plt.title("The number of Employees", fontsize = 20)
plt.show()
# 해가 지날수록 근로자 수가 서서히 증가함을 알 수 있다.
# %%
#근로자수와 식수 관계 파악
fig, ax = plt.subplots(figsize=(20,10))
sns.regplot(x='employees', y = 'num_lunch', data = train_data, color = 'salmon', marker = '+')
sns.regplot(x='employees', y = 'num_dinner', data = train_data, color = 'blue', marker = 'o')
# 근로자수가 증가하면 식수가 감소함을 확인했다.
# 그러나 데이터의 분포를 보니 관계파악이 힘들며 이를 통해 근로자 수와 식수와의 관련 없다고 판단했다.
# %%

# %%

# %%
