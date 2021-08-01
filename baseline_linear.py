# %%

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

# %%
df_orig_train = pd.read_csv('./preprocessed_data/preprocessed_train.csv')
df_orig_test = pd.read_csv('./preprocessed_data/preprocessed_test.csv')
submission = pd.read_csv('./data/sample_submission.csv')

## 중식예측

linear_lunch = LinearRegression(n_jobs=-1)
linear_lunch.fit(df_orig_train[['식사대상자', '요일_lunch', '본사시간외근무명령서승인건수',]], df_orig_train['중식계'])

print(mean_absolute_error(df_orig_train['중식계'], linear_lunch.predict(df_orig_train[['식사대상자', '요일_lunch', '본사시간외근무명령서승인건수',]])))

## 석식예측

linear_dinner = LinearRegression(n_jobs=-1)
linear_dinner.fit(df_orig_train[['식사대상자', '본사시간외근무명령서승인건수', '요일_dinner']], df_orig_train['석식계'])

print(mean_absolute_error(df_orig_train['석식계'], linear_dinner.predict(df_orig_train[['식사대상자', '본사시간외근무명령서승인건수', '요일_dinner']])))
# %%
lunch_pred = linear_lunch.predict(df_orig_test[['식사대상자', '요일_lunch', '본사시간외근무명령서승인건수',]])
dinner_pred = linear_dinner.predict(df_orig_test[['식사대상자', '본사시간외근무명령서승인건수', '요일_dinner']])

submission['중식계'] = lunch_pred
submission['석식계'] = dinner_pred

submission.to_csv('linear_model_add_feature.csv', index=False)
# %%
# %%
