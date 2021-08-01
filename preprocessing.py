# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
pd.set_option("display.max_columns", None)
# %%
df_orig_train = pd.read_csv('./data/train.csv')
df_orig_test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sample_submission.csv')

# %%
date_mapper = {'월': 'Monday', '화':'Tuesday', '수':'Wednesday', '목':'Thursday', '금':'Friday', '토':'Saturday', '일':'Sunday'}
df_orig_train['요일'] = df_orig_train['요일'].map(date_mapper)
df_orig_test['요일'] = df_orig_test['요일'].map(date_mapper)

# %%
import re
predefined_patten = r"\s*[(<]\s*(\w+[,]*)*\w*\s*[:]*\s*[/]*(\w+[,]*)*\s*[:]*\s*(\w+[,]*)*\s*[)>]\s*"

df_orig_train['중식메뉴_processed'] = df_orig_train['중식메뉴'].apply(lambda x: re.sub(predefined_patten, " ", x))\
    .apply(lambda x: re.sub("\s+", " ", x))

df_orig_train['석식메뉴_processed'] = df_orig_train['석식메뉴'].apply(lambda x: re.sub(predefined_patten, " ", x))\
    .apply(lambda x: re.sub("\s+", " ", x))
    
df_orig_test['중식메뉴_processed'] = df_orig_test['중식메뉴'].apply(lambda x: re.sub(predefined_patten, " ", x))\
    .apply(lambda x: re.sub("\s+", " ", x))

df_orig_test['석식메뉴_processed'] = df_orig_test['석식메뉴'].apply(lambda x: re.sub(predefined_patten, " ", x))\
    .apply(lambda x: re.sub("\s+", " ", x))

# %%
df_orig_train['일자'] = df_orig_train['일자'].astype('datetime64')
df_orig_test['일자'] = df_orig_test['일자'].astype('datetime64')

df_orig_train['Year'] = df_orig_train['일자'].dt.year
df_orig_train['Month'] = df_orig_train['일자'].dt.month
df_orig_train['Day'] = df_orig_train['일자'].dt.day

df_orig_test['Year'] = df_orig_test['일자'].dt.year
df_orig_test['Month'] = df_orig_test['일자'].dt.month
df_orig_test['Day'] = df_orig_test['일자'].dt.day

df_orig_train = df_orig_train.assign(식사대상자 = lambda x: x['본사정원수'] - x['본사휴가자수'] - x['본사출장자수'] - x['현본사소속재택근무자수'])
df_orig_test = df_orig_test.assign(식사대상자 = lambda x: x['본사정원수'] - x['본사휴가자수'] - x['본사출장자수'] - x['현본사소속재택근무자수'])

# %%

tmp_lunch = df_orig_train.groupby(['Month', '요일'])['중식계'].apply(np.median).reset_index(name='month_days_lunch')
tmp_dinner = df_orig_train.groupby(['Month', '요일'])['석식계'].apply(np.median).reset_index(name='month_days_dinner')

df_orig_train = pd.merge(df_orig_train, tmp_lunch, how='left', 
                         left_on=['Month', '요일'], 
                         right_on=['Month', '요일'])

df_orig_train = pd.merge(df_orig_train, tmp_dinner, how='left', 
                         left_on=['Month', '요일'], 
                         right_on=['Month', '요일'])

df_orig_test = pd.merge(df_orig_test, tmp_lunch, how='left', 
                        left_on=['Month', '요일'], 
                        right_on=['Month', '요일'])


df_orig_test = pd.merge(df_orig_test, tmp_dinner, how='left', 
                        left_on=['Month', '요일'], 
                        right_on=['Month', '요일'])


df_orig_train['요일_lunch'] = df_orig_train['요일'].map(dict(df_orig_train.groupby(['요일'])['중식계'].apply(np.median)))
df_orig_test['요일_lunch'] = df_orig_test['요일'].map(dict(df_orig_train.groupby(['요일'])['중식계'].apply(np.median)))

df_orig_train['요일_dinner'] = df_orig_train['요일'].map(dict(df_orig_train.groupby(['요일'])['석식계'].apply(np.median)))
df_orig_test['요일_dinner'] = df_orig_test['요일'].map(dict(df_orig_train.groupby(['요일'])['석식계'].apply(np.median)))

# %%

df_orig_train = df_orig_train.assign(점심_변동성 = lambda x: x['중식계'] - x['요일_lunch'])\
    .assign(석식_변동성 = lambda x: x['석식계'] - x['요일_dinner'])
df_orig_train = df_orig_train.assign(점심_변동율 = lambda x: (x['중식계'] - x['요일_lunch']) / x['식사대상자'])\
    .assign(석식_변동율 = lambda x: (x['석식계'] - x['요일_dinner']) / x['식사대상자'])

df_orig_train['covid'] = np.where(df_orig_train['현본사소속재택근무자수'] >= 1, 1, 0)
df_orig_test['covid'] = np.where(df_orig_test['현본사소속재택근무자수'] >= 1, 1, 0)
# %%
df_orig_train.to_csv('./preprocessed_data/preprocessed_train.csv', index=False)
df_orig_test.to_csv('./preprocessed_data/preprocessed_test.csv', index=False)

# %%
