import pandas as pd
import numpy as np
import re

def preprocess_date(data):
    """
    영어로 표시된 요일을 한글로 변환.
    일자를 Year, Month, Day로 분리.

    Parameters:
    data(dtype=DataFrame): data
    
    Returns:
    data(dtype=DataFrame): data
    """
    # 영어로 표시된 요일을 한글로 바꾼다.
    date_mapper = {'월': 'Monday', '화': 'Tuesday', '수': 'Wednesday', 
                   '목': 'Thursday', '금': 'Friday', '토': 'Saturday', '일': 'Sunday'}
        
    data['요일'] = data['요일'].map(date_mapper)

    # 일자를 Year, Month, Day로 분리한다. 
    data['일자'] = data['일자'].astype('datetime64')
    data['Year'] = data['일자'].dt.year
    data['Month'] = data['일자'].dt.month
    data['Day'] = data['일자'].dt.day

    return data


def preprocess_menu(data):
    """
    중식메뉴, 석식메뉴 이름만 띄어쓰기로 분리하여 표시.

    Parameters:
    data(dtype=DataFrame): data
    
    Returns:
    data(dtype=DataFrame): data
    """
    predefined_patten = r"\s*[(<]\s*(\w+[,]*)*\w*\s*[:]*\s*[/]*(\w+[,]*)*\s*[:]*\s*(\w+[,]*)*\s*[)>]\s*"

    data['중식메뉴_processed'] = data['중식메뉴']\
        .apply(lambda x: re.sub(predefined_patten, " ", x))\
        .apply(lambda x: re.sub("\s+", " ", x))

    data['석식메뉴_processed'] = data['석식메뉴']\
        .apply(lambda x: re.sub(predefined_patten, " ", x))\
        .apply(lambda x: re.sub("\s+", " ", x))
    
    return data


def feature_people(data):
    """
    잠재적인 식사대상자를 게산하여 feature로 추가.

    Parameters:
    data(dtype=DataFrame): data
    
    Returns:
    data(dtype=DataFrame): data
    """
    data = data.assign(식사대상자 = lambda x:\
        x['본사정원수'] - x['본사휴가자수'] - x['본사출장자수'] - x['현본사소속재택근무자수'])

    return data

def feature_meal(data, med_lunch, med_dinner):
    """
    매월 요일에 따른 중식계, 석식계 중앙값 계산.
    매 요일 중식계, 석식계 중앙값 계산.

    Parameters:
    data(dtype=DataFrame): data
    med_lunch(dtype=DataFrame): 매월 요일에 따른 중식계 중앙값
    med_dinner(dtype=DataFrame): 매월 요일에 따른 석식계 중앙값
    
    Returns:
    data(dtype=DataFrame): data  
    """
    data = pd.merge(data, med_lunch, how='left', 
                    left_on=['Month', '요일'], 
                    right_on=['Month', '요일'])
    data = pd.merge(data, med_dinner, how='left', 
                    left_on=['Month', '요일'], 
                    right_on=['Month', '요일'])

    return data


def change_rate(data):
    """
    잠재적인 식사대상자와 실제 식수와 비교하여 변동성과 변동율 계산.

    Parameters:
    data(dtype=DataFrame): data
    
    Returns:
    data(dtype=DataFrame): data  
    """
    data = data.assign(점심_변동성 = lambda x: x['중식계'] - x['요일_lunch'])\
        .assign(석식_변동성 = lambda x: x['석식계'] - x['요일_dinner'])
    data = data.assign(점심_변동율 = lambda x: (x['중식계'] - x['요일_lunch']) / x['식사대상자'])\
        .assign(석식_변동율 = lambda x: (x['석식계'] - x['요일_dinner']) / x['식사대상자'])
    
    return data


def feature_covid(data):
    """
    재택근무자수 유무에 따라서 covid feature 생성.

    Parameters:
    data(dtype=DataFrame): data
    
    Returns:
    data(dtype=DataFrame): data  
    """
    data['covid'] = np.where(data['현본사소속재택근무자수'] >= 1, 1, 0)

    return data


class Get_data(object):
    def __init__(self, args):
        self.args = args

    def preprocess(self, train_data, test_data):
        train_data = preprocess_date(train_data)
        train_data = preprocess_menu(train_data)

        test_data = preprocess_date(test_data)
        test_data = preprocess_menu(test_data)
        
        return train_data, test_data


    def feature_engineering(self, train_data, test_data):
        train_data = feature_people(train_data)
        test_data = feature_people(test_data)

        med_lunch = train_data.groupby(['Month', '요일'])['중식계']\
            .apply(np.median).reset_index(name='month_days_lunch')
        med_dinner = train_data.groupby(['Month', '요일'])['석식계']\
            .apply(np.median).reset_index(name='month_days_dinner')

        train_data = feature_meal(train_data, med_lunch, med_dinner)
        train_data = feature_covid(train_data)

        test_data = feature_meal(test_data, med_lunch, med_dinner)
        test_data = feature_covid(test_data)

        train_data['요일_lunch'] = train_data['요일'].\
            map(dict(train_data.groupby(['요일'])['중식계'].apply(np.median)))
        train_data['요일_dinner'] = train_data['요일'].\
            map(dict(train_data.groupby(['요일'])['석식계'].apply(np.median)))
        test_data['요일_lunch'] = test_data['요일'].\
            map(dict(train_data.groupby(['요일'])['중식계'].apply(np.median)))
        test_data['요일_dinner'] = test_data['요일'].\
            map(dict(train_data.groupby(['요일'])['석식계'].apply(np.median)))
        
        return train_data, test_data 
