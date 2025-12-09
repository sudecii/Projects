from fastapi import FastAPI
import joblib
import lightgbm
import pandas as pd
from pydantic import BaseModel
from datetime import datetime
import numpy as np

# объект класса фастапи
app = FastAPI()

# Проверяем работу сервиса
@app.get('/')
def init():
    return 'service is working'

class ClientData(BaseModel):
    # понятные числовые
    age: int
    income: float              
    region_rating: int
    appl_rej_cnt: int
    out_request_cnt: int
    Score_bki: float

    first_time_cd: int         # 0/1
    good_work_flg: int         # 0/1
    SNA: int                   # 0/1

    # категориальные
    gender_cd: str
    education_cd: str
    car_own_flg: str
    car_type_flg: str
    home_address_cd: int
    work_address_cd: int
    Air_flg: str

# Столбцы в правильном порядке
FEATURE_COLUMNS = [
    'education_cd',
    'gender_cd',
    'age',
    'car_own_flg',
    'car_type_flg',
    'appl_rej_cnt',
    'good_work_flg',
    'Score_bki',
    'out_request_cnt',
    'region_rating',
    'home_address_cd',
    'work_address_cd',
    'SNA',
    'first_time_cd',
    'Air_flg',
    'app_month',
    'app_dayofweek',
    'income_log',
    'income_per_region_log',
    'age_per_rejcnt'
]

# Функция для проектирования признаков
def build_features(data):
    now = datetime.now()

    row = {
        # всё, что и так есть у пользователя
        'education_cd':   data.education_cd,
        'gender_cd':      data.gender_cd,
        'age':            data.age,
        'car_own_flg':    data.car_own_flg,
        'car_type_flg':   data.car_type_flg,
        'appl_rej_cnt':   data.appl_rej_cnt,
        'good_work_flg':  data.good_work_flg,
        'Score_bki':      data.Score_bki,
        'out_request_cnt': data.out_request_cnt,
        'region_rating':  data.region_rating,
        'home_address_cd': data.home_address_cd,
        'work_address_cd': data.work_address_cd,
        'SNA':            data.SNA,
        'first_time_cd':  data.first_time_cd,
        'Air_flg':        data.Air_flg,

        # календарные фичи – из текущей даты
        'app_month':     now.month,
        'app_dayofweek': now.weekday(),
    }

    # инженерные фичи – должны совпадать с тем, как ты делал в ноутбуке
    row['income_log'] = np.log1p(data.income)
    row['income_per_region_log'] = np.log1p(data.income / max(data.region_rating, 1))
    row['age_per_rejcnt'] = data.age / (1 + data.appl_rej_cnt)

    # форфимурем датафрейм с правильными колнками и фичами
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return df
    
THRESHOLD = 0.19  # оптимальный порог по критерию Юдена (из ноутбука тренировочного)

model_pipe = joblib.load('model.pkl')

# Обработчик с предиктом скора
@app.post('/predict')
def predict(client_data: ClientData):
    # Матрица признаков
    X = build_features(client_data)
    # вероятность дефолта
    y_proba_pred = model_pipe.predict_proba(X)[0, 1]
    # Одобирил если меньше порога
    approved = bool(y_proba_pred < THRESHOLD)

    # возвращаем статус и вероятность дефолта
    return {
        'default_proba': float(y_proba_pred),
        'approved': approved
    }