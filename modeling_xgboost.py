# %%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
pd.set_option("display.max_columns", None)

# %%
from sklearn.model_selection import train_test_split, ParameterGrid, KFold
from tqdm import tqdm

def grid_search_model(train_data, parameter_candidate, basic_param, n_splits, feature_mask, label):
    
    results = pd.DataFrame([], columns=['parameter', 'loss'])
    
    for parameter in tqdm(ParameterGrid(parameter_candidate), desc='combination'):
        
        parameter.update(basic_param)
        
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=1004)
        
        tmp_results = []
        tmp_param = []
        
        for train_index, test_index in skf.split(train_data[feature_mask], train_data[label]):
            
            if label == '석식계':
                X_train, X_test = train_data.iloc[train_index][feature_mask], train_data.iloc[test_index][feature_mask]
                y_train, y_test = train_data.iloc[train_index][label], train_data.iloc[test_index][label]
                X_train, X_test = X_train[y_train != 0], X_test[y_test != 0]
                y_train, y_test = y_train[y_train != 0], y_test[y_test != 0]
                
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
                
            else:
                X_train, X_test = train_data.iloc[train_index][feature_mask], train_data.iloc[test_index][feature_mask]
                y_train, y_test = train_data.iloc[train_index][label], train_data.iloc[test_index][label]
                
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_test, label=y_test)
            
            xgb_model = xgb.train(dtrain=dtrain, params=parameter, evals=[(dtest, 'validation')], 
                                  verbose_eval=0, early_stopping_rounds=500, num_boost_round=10000)
            
            tmp_results.append(mean_absolute_error(y_pred=xgb_model.predict(dtest), y_true=y_test))
            tmp_param.append(str(parameter))
            
        results = results.append(pd.DataFrame({"parameter":tmp_param, "loss":tmp_results}))
    
    return results

# %%

if __name__ == "__main__":
    
    df_train = pd.read_csv("./preprocessed_data/preprocessed_train.csv")
    df_test = pd.read_csv("./preprocessed_data/preprocessed_test.csv")
    submission = pd.read_csv('./data/sample_submission.csv')
    
    feature_mask_lunch = ['본사정원수', '본사휴가자수', '본사출장자수', '현본사소속재택근무자수', '식사대상자', '요일_lunch', '요일_dinner', '본사시간외근무명령서승인건수', 'covid', 'Year', 'Month', 'month_days_lunch', 'month_days_dinner']
    feature_mask_dinner = ['본사정원수', '본사휴가자수', '본사출장자수', '현본사소속재택근무자수', '식사대상자', '요일_lunch', '요일_dinner', '본사시간외근무명령서승인건수', 'covid', 'Year', 'Month', 'month_days_lunch', 'month_days_dinner']
    
    parameter_grid = {'learning_rate':[0.5, 0.01, 0.001], 
                      'max_depth':[3, 4, 5, 6], 
                      'lambda':[1, 2 ,3], 
                      'gamma':[0, 0.1, 0.2, 0.3]}
    
    basic_parameter = {'objective':'reg:squarederror', 
                       'eval_metric':'rmse', 
                       'tree_method':'gpu_hist', 
                       'gpu_id':'0'}
    
    
    xgboost_results_lunch = grid_search_model(train_data=df_train,
                                        parameter_candidate=parameter_grid, 
                                        basic_param=basic_parameter, 
                                        n_splits=5, 
                                        feature_mask=feature_mask_lunch, 
                                        label='중식계')
    
    xgboost_results_dinner = grid_search_model(train_data=df_train,
                                        parameter_candidate=parameter_grid, 
                                        basic_param=basic_parameter, 
                                        n_splits=5, 
                                        feature_mask=feature_mask_lunch, 
                                        label='석식계')

    xgboost_lunch_results = xgboost_results_lunch.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    xgboost_lunch_results['type'] = 'lunch'
    xgboost_lunch_results['model'] = 'xgboost'


    xgboost_dinner_results = xgboost_results_dinner.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    xgboost_dinner_results['type'] = 'dinner'
    xgboost_dinner_results['model'] = 'xgboost'

    results = pd.concat([xgboost_lunch_results, xgboost_dinner_results], axis=0)

    results.to_excel("./result/xgboost_results_add.xlsx", index=False)

# %%
