# 구내식당 식수 인원 예측 AI 경진대회

구내식당의 요일별 점심, 저녁식사를 먹는 인원을 예측 ([homepage](https://dacon.io/competitions/official/235743/overview/description))

주최 : 한국토지주택공사

주관 : 데이콘

</br>

## Schematic diagram
<p align="center">
  <img width="500" alt="Schematic_diagram" src="https://user-images.githubusercontent.com/68064510/220250399-82d5c73e-173c-4203-851b-b790dc63f7fa.png">

</br>

## 1. Requirements
```
pip install -r requirements.txt
```

## 2. Grid search
xgboost, LinearRegression, ElasticNet, RandomForestRegressor의 grid search를 통해 MSE가 가장 낮은 모델과 해당 하이퍼파라미터를 파악할 수 있습니다.

```
python3 modeling.py  
```

## 3. Submit
grid search 실험을 통해 최종 선택된 모델을 동작시키는 코드입니다.
```
python3 submit.py
```