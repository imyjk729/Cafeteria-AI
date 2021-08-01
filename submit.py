# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
pd.set_option("display.max_columns", None)

# %%
from sklearn.model_selection import train_test_split

# %%

if __name__ == "__main__":
    
    df_train = pd.read_csv("./preprocessed_data/preprocessed_train.csv")
    df_test = pd.read_csv("./preprocessed_data/preprocessed_test.csv")
    submission = pd.read_csv('./data/sample_submission.csv')
    
    feature_mask_lunch = ['식사대상자', '요일_lunch', '본사시간외근무명령서승인건수', 'covid']
    feature_mask_dinner = ['식사대상자', '요일_dinner', '본사시간외근무명령서승인건수', 'covid']
    
    feature_mask_lunch = ['본사정원수', '본사휴가자수', '본사출장자수', '현본사소속재택근무자수', '식사대상자', '요일_lunch', '요일_dinner', '본사시간외근무명령서승인건수', 'covid']
    feature_mask_dinner = ['본사정원수', '본사휴가자수', '본사출장자수', '현본사소속재택근무자수', '식사대상자', '요일_lunch', '요일_dinner', '본사시간외근무명령서승인건수', 'covid']
    
    parameter_grid = {'learning_rate':0.01, 
                      'max_depth':3, 
                      'lambda':1, 
                      'gamma':0, 
                      'objective':'reg:squarederror', 
                       'eval_metric':'rmse', 
                       'tree_method':'gpu_hist', 
                       'gpu_id':'0'}
    
    trainset, testset = train_test_split(df_train, train_size=0.8, random_state=1004)
    
    dtrain_lunch = xgb.DMatrix(trainset[feature_mask_lunch], label=trainset['중식계'])
    dtest_lunch = xgb.DMatrix(testset[feature_mask_lunch], label=testset['중식계'])
    
    xgb_model_lunch = xgb.train(dtrain=dtrain_lunch, params=parameter_grid, evals=[(dtest_lunch, 'validation')], 
                                verbose_eval=0, early_stopping_rounds=500, num_boost_round=2000)
    
    dtrain_dinner = xgb.DMatrix(trainset[trainset['석식계'] != 0][feature_mask_dinner], label=trainset[trainset['석식계'] != 0]['석식계'])
    dtest_dinner = xgb.DMatrix(testset[testset['석식계'] != 0][feature_mask_dinner], label=testset[testset['석식계'] != 0]['석식계'])
    
    parameter_grid = {'learning_rate':0.01, 
                      'max_depth':4, 
                      'lambda':2, 
                      'gamma':0, 
                      'objective':'reg:squarederror', 
                      'eval_metric':'rmse', 
                      'tree_method':'gpu_hist', 
                      'gpu_id':'0'}
    
    xgb_model_dinner = xgb.train(dtrain=dtrain_dinner, params=parameter_grid, evals=[(dtest_dinner, 'validation')], 
                                verbose_eval=0, early_stopping_rounds=500, num_boost_round=2000)
    
    lunch = xgb_model_lunch.predict(xgb.DMatrix(df_test[feature_mask_lunch]))
    dinner = xgb_model_dinner.predict(xgb.DMatrix(df_test[feature_mask_dinner]))
    
    submission['중식계'] = lunch
    submission['석식계'] = dinner

    submission.to_csv('./submit/xgboost_results.csv', index=False)

# %%
