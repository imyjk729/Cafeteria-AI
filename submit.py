import argparse
import pandas as pd
import numpy as np
import xgboost as xgb

from dataset import Get_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def define_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        default='./data/train.csv',
        help="Train data path"
    )
    parser.add_argument(
        "--test_file",
        default='./data/test.csv',
        help="Test data path"
    )
    parser.add_argument(
        "--out_file",
        default='./data/sample_submission.csv',
        help="output file path"
    )

    parser.add_argument(
        "--feature_mask",
        default=[
            '본사정원수', '본사휴가자수', '본사출장자수', '현본사소속재택근무자수',
            '식사대상자', '요일_lunch', '요일_dinner', '본사시간외근무명령서승인건수', 'covid'
        ],
        nargs='*',
        type=list,
        help="Features list to use"
    )
    parser.add_argument(
        "--param_lunch",
        default={
            'learning_rate': 0.01,
            'max_depth': 3,
            'lambda': 1,
            'gamma': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist',
            'gpu_id': '0'
        },
        nargs='+',
        type=dict,
        help="Paramer dictionary for lunch"
    )
    parser.add_argument(
        "--param_dinner",
        default={
            'learning_rate': 0.01,
            'max_depth': 4,
            'lambda': 2,
            'gamma': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist',
            'gpu_id': '0'
        },
        nargs='+',
        type=dict,
        help="Paramer dictionary for dinner"
    )

    # Set parameters
    args = parser.parse_args()

    return args


def main(args):
    # Load data
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)
    submission = pd.read_csv(args.out_file)

    get_data = Get_data(args)
    df_train, df_test = get_data.preprocess(train_df, test_df)
    df_train, df_test = get_data.feature_engineering(df_train, df_test)

    trainset, testset = train_test_split(
        df_train, train_size=0.8, random_state=1004)

    # Modeling
    dtrain_lunch = xgb.DMatrix(
        trainset[args.feature_mask],
        label=trainset['중식계']
    )
    dtest_lunch = xgb.DMatrix(
        testset[args.feature_mask],
        label=testset['중식계']
    )

    xgb_model_lunch = xgb.train(
        dtrain=dtrain_lunch,
        params=args.param_lunch,
        evals=[(dtest_lunch, 'validation')],
        verbose_eval=0,
        early_stopping_rounds=500,
        num_boost_round=2000
    )

    dtrain_dinner = xgb.DMatrix(
        trainset[trainset['석식계'] != 0][args.feature_mask],
        label=trainset[trainset['석식계'] != 0]['석식계']
    )
    dtest_dinner = xgb.DMatrix(
        testset[testset['석식계'] != 0][args.feature_mask],
        label=testset[testset['석식계'] != 0]['석식계']
    )

    xgb_model_dinner = xgb.train(
        dtrain=dtrain_dinner,
        params=args.param_dinner,
        evals=[(dtest_dinner, 'validation')],
        verbose_eval=0,
        early_stopping_rounds=500,
        num_boost_round=2000
    )

    lunch = xgb_model_lunch.predict(xgb.DMatrix(df_test[args.feature_mask]))
    dinner = xgb_model_dinner.predict(xgb.DMatrix(df_test[args.feature_mask]))

    submission['중식계'] = lunch
    submission['석식계'] = dinner

    submission.to_csv('./results.csv', index=False)


if __name__ == "__main__":
    args = define_argparser()
    main(args)
