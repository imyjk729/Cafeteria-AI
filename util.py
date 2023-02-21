from sklearn.model_selection import ParameterGrid, KFold
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import pandas as pd
import xgboost


def grid_search(train_data, parameter_candidate, model_name, basic_param, n_splits, feature_mask, label):
    
    results = pd.DataFrame([], columns=['parameter', 'loss'])
    
    for parameter in tqdm(ParameterGrid(parameter_candidate), desc='combination'):
        
        if model_name == xgboost:
            parameter.update(basic_param)
        else:
            model = model_name(**parameter)
        
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
            
        if model_name == xgboost:
            dtrain = xgboost.DMatrix(X_train, label=y_train)
            dtest = xgboost.DMatrix(X_test, label=y_test) 
            xgb_model = xgboost.train(dtrain=dtrain, params=parameter, evals=[(dtest, 'validation')], 
                                  verbose_eval=0, early_stopping_rounds=500, num_boost_round=10000)  
            tmp_results.append(mean_absolute_error(y_pred=xgb_model.predict(dtest), y_true=y_test))          
        else:
            model.fit(X_train, y_train)
            tmp_results.append(mean_absolute_error(y_pred=model.predict(X_test), y_true=y_test))
            
        tmp_param.append(str(parameter))    
        results = results.append(pd.DataFrame({"parameter":tmp_param, "loss":tmp_results}))

    return results

