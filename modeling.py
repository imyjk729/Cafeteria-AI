import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
import xgboost
from util import grid_search
from dataset import Get_data


def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", 
        default='./data/train.csv', 
        help="Train data path")
    parser.add_argument("--test_file", 
        default='./data/test.csv', 
        help="Test data path")

    parser.add_argument("--features_xgb", 
        default=['본사정원수', '본사휴가자수', '본사출장자수', '현본사소속재택근무자수', '식사대상자',\
                 '요일_lunch', '요일_dinner', '본사시간외근무명령서승인건수', 'covid',\
                 'Year', 'Month', 'month_days_lunch', 'month_days_dinner'], 
        nargs='*', type=list, help="Features list for xgboost")
    parser.add_argument("--fixed_param_xgb", 
        default={'objective':'reg:squarederror', 
                 'eval_metric':'rmse', 
                 'tree_method':'gpu_hist', 
                 'gpu_id':'0'}, 
        nargs='+', type=dict, help="Fixed parameter dictionary for xgboost")
    parser.add_argument("--param_grid_xgb", 
        default={'learning_rate':[0.5, 0.01, 0.001], 
                 'max_depth':[3, 4, 5, 6], 
                 'lambda':[1, 2 ,3], 
                 'gamma':[0, 0.1, 0.2, 0.3]}, 
        nargs='+', type=dict, help="Paramer dictionary to apply grid search for xgboost")

    parser.add_argument("--feature_lunch_model", 
        default=['식사대상자', '요일_lunch', '본사시간외근무명령서승인건수', 'covid'], 
        nargs='*', type=list, help="Features list to use")
    parser.add_argument("--feature_dinner_model", 
        default=['식사대상자', '요일_dinner', '본사시간외근무명령서승인건수', 'covid'], 
        nargs='*', type=list, help="Features list to use")
    parser.add_argument("--models_list", 
        default=[RandomForestRegressor, LinearRegression, ElasticNet], 
        nargs='*', type=list, help="Models list")
      
    parser.add_argument("--rf_param", 
        default={'n_jobs':[-1], 
                 'n_estimators':[10, 50, 100, 150, 200, 250, 300, 350, 400], 
                 'max_depth':[2, 3, 4, None]}, 
        nargs='+', type=dict, help="Dictionary of RandomForest regression parameters")
    parser.add_argument("--linear_param", 
        default={'fit_intercept':[True, False], 
                 'normalize':[True, False]}, 
        nargs='+', type=dict, help="Dictionary of Linear regression parameters")
    parser.add_argument("--elastic_param", 
        default={'alpha':range(0, 2, 20), 
                 'l1_ratio':range(0, 1, 10), 
                 'normalize':[True, False], 
                 'fit_intercept':[True, False]},
        nargs='+', type=dict, help="Dictionary of ElasticNet parameters")

    # Set parameters
    args = parser.parse_args()

    return args


def main(args):
    # Load data
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)

    get_data = Get_data(args)
    train_data, test_data = get_data.preprocess(train_df, test_df)
    train_data, test_data = get_data.feature_engineering(train_data, test_data)

    xgboost_results_lunch = grid_search(train_data=train_data,
                                        parameter_candidate=args.param_grid_xgb, 
                                        model_name=xgboost,
                                        basic_param=args.fixed_param_xgb, 
                                        n_splits=5, 
                                        feature_mask=args.features_xgb, 
                                        label='중식계')
    
    xgboost_results_dinner = grid_search(train_data=train_data,
                                        parameter_candidate=args.param_grid_xgb, 
                                        model_name=xgboost,
                                        basic_param=args.fixed_param_xgb, 
                                        n_splits=5, 
                                        feature_mask=args.features_xgb, 
                                        label='석식계')

    xgboost_lunch_results = xgboost_results_lunch.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    xgboost_lunch_results['type'] = 'lunch'
    xgboost_lunch_results['model'] = xgboost.__name__


    xgboost_dinner_results = xgboost_results_dinner.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
    xgboost_dinner_results['type'] = 'dinner'
    xgboost_dinner_results['model'] = xgboost.__name__

    results = pd.concat([xgboost_lunch_results, xgboost_dinner_results], axis=0)

    # Apply grid search and proceed with modeling.
    parameters_list = [args.rf_param, args.linear_param, args.elastic_param]

    for i in range(len(args.models_list)):
        results_lunch = grid_search(train_data=train_data,
                                   parameter_candidate=parameters_list[i],
                                   model_name=args.models_list[i],
                                   basic_param=args.fixed_param_xgb,
                                   n_splits=5,
                                   feature_mask=args.feature_lunch_model, 
                                   label='중식계')
        results_dinner = grid_search(train_data=train_data,
                                    parameter_candidate=parameters_list[i],
                                    model_name=args.models_list[i],
                                    basic_param=args.fixed_param_xgb,
                                    n_splits=5,
                                    feature_mask=args.feature_dinner_model, 
                                    label='석식계')

        lunch_results = results_lunch.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
        lunch_results['type'] = 'lunch'
        lunch_results['model'] = args.models_list[i].__name__

        dinner_results = results_dinner.groupby(['parameter'])['loss'].apply(np.mean).reset_index(name='loss')
        dinner_results['type'] = 'dinner'
        dinner_results['model'] = args.models_list[i].__name__

        results = pd.concat([results, lunch_results, dinner_results], axis=0)

    results.to_excel("./ml_results.xlsx", index=False)


if __name__ == "__main__":
    args = define_argparser()
    main(args)
