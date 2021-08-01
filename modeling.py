# %%
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)

# %%
from sklearn.model_selection import ParameterGrid, KFold
from tqdm import tqdm

def grid_search_model(train_data, parameter_candidate, model_class, n_splits, feature_mask, label):
    
    results = pd.DataFrame([], columns=['parameter', 'loss'])
    
    for parameter in tqdm(ParameterGrid(parameter_candidate), desc='combination'):
        
        model = model_class(**parameter)
        
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=1004)
        
        tmp_results = []
        tmp_param = []
        
        for train_index, test_index in skf.split(train_data[feature_mask], train_data[label]):
            
            if label == '석식계':
                X_train, X_test = train_data.iloc[train_index][feature_mask], train_data.iloc[test_index][feature_mask]
                y_train, y_test = train_data.iloc[train_index][label], train_data.iloc[test_index][label]
                X_train, X_test = X_train[y_train != 0], X_test[y_test != 0]
                y_train, y_test = y_train[y_train != 0], y_test[y_test != 0]
            else:
                X_train, X_test = train_data.iloc[train_index][feature_mask], train_data.iloc[test_index][feature_mask]
                y_train, y_test = train_data.iloc[train_index][label], train_data.iloc[test_index][label]
            
            model.fit(X_train, y_train)
            
            tmp_results.append(mean_absolute_error(y_pred=model.predict(X_test), y_true=y_test))
            tmp_param.append(str(parameter))
            
        results = results.append(pd.DataFrame({"parameter":tmp_param, "loss":tmp_results}))
    
    return results

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    
    df_train = pd.read_csv("./preprocessed_data/preprocessed_train.csv")
    df_test = pd.read_csv("./preprocessed_data/preprocessed_test.csv")
    submission = pd.read_csv('./data/sample_submission.csv')
    
    feature_mask_lunch = ['식사대상자', '요일_lunch', '본사시간외근무명령서승인건수', 'covid']
    feature_mask_dinner = ['식사대상자', '요일_dinner', '본사시간외근무명령서승인건수', 'covid']
    
    rf_parameters = {'n_jobs':[-1], 
                    'n_estimators':[10, 50, 100, 150, 200, 250, 300, 350, 400], 
                    "max_depth":[2, 3, 4, None]}
    
    rf_results_lunch = grid_search_model(train_data=df_train, 
                                        parameter_candidate=rf_parameters, 
                                        model_class=RandomForestRegressor, 
                                        n_splits=5, 
                                        feature_mask=feature_mask_lunch, 
                                        label='중식계')
    
    rf_results_dinner = grid_search_model(train_data=df_train, 
                                        parameter_candidate=rf_parameters, 
                                        model_class=RandomForestRegressor, 
                                        n_splits=5, 
                                        feature_mask=feature_mask_dinner, 
                                        label='석식계')
    
    linear_parameters = {'fit_intercept':[True, False], 
                        'normalize':[True, False]}
    
    linear_results_lunch = grid_search_model(train_data=df_train, 
                                        parameter_candidate=linear_parameters, 
                                        model_class=LinearRegression, 
                                        n_splits=5, 
                                        feature_mask=feature_mask_lunch, 
                                        label='중식계')
    
    linear_results_dinner = grid_search_model(train_data=df_train, 
                                        parameter_candidate=linear_parameters, 
                                        model_class=LinearRegression, 
                                        n_splits=5, 
                                        feature_mask=feature_mask_dinner, 
                                        label='석식계')
    
    elastic_parameters = {'alpha':range(0, 2, 20), 
                        'l1_ratio':range(0, 1, 10), 
                        "normalize":[True, False], 
                        'fit_intercept':[True, False]}
    
    elastic_results_lunch = grid_search_model(train_data=df_train, 
                                        parameter_candidate=elastic_parameters, 
                                        model_class=ElasticNet, 
                                        n_splits=5, 
                                        feature_mask=feature_mask_lunch, 
                                        label='중식계')
    
    elastic_results_dinner = grid_search_model(train_data=df_train, 
                                        parameter_candidate=elastic_parameters, 
                                        model_class=ElasticNet, 
                                        n_splits=5, 
                                        feature_mask=feature_mask_dinner, 
                                        label='석식계')
    
    rf_lunch_results = rf_results_lunch.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    rf_lunch_results['type'] = 'lunch'
    rf_lunch_results['model'] = 'rf'

    rf_dinner_results = rf_results_dinner.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    rf_dinner_results['type'] = 'dinner'
    rf_dinner_results['model'] = 'rf'

    linear_lunch_results = linear_results_lunch.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    linear_lunch_results['type'] = 'lunch'
    linear_lunch_results['model'] = 'linear'

    linear_dinner_results = linear_results_dinner.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    linear_dinner_results['type'] = 'dinner'
    linear_dinner_results['model'] = 'linear'

    elastic_lunch_results = elastic_results_lunch.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    elastic_lunch_results['type'] = 'lunch'
    elastic_lunch_results['model'] = 'elastic'

    elastic_dinner_results = elastic_results_dinner.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    elastic_dinner_results['type'] = 'dinner'
    elastic_dinner_results['model'] = 'elastic'

    results = pd.concat([rf_lunch_results, rf_dinner_results, 
                        linear_lunch_results, linear_dinner_results, 
                        elastic_lunch_results, elastic_dinner_results], axis=0)
    results.to_excel("./result/ml_results.xlsx", index=False)