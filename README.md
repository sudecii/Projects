# 🚀 ML Projects Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-lightgrey.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Models-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Data Science](https://img.shields.io/badge/Data%20Science-EDA%20%7C%20Visualization-yellow.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

---

Добро пожаловать в мой репозиторий проектов по анализу данных и машинному обучению!  
Здесь собраны работы, демонстрирующие навыки в области **EDA, моделирования, продакшен-интеграции и визуализации данных**.  
Каждый проект оформлен в отдельной папке с описанием, кодом и результатами.

---


## 🧩 Проекты

### 🏡 [Housing_Price_Prediction](./housing-price-prediction)

💻 **Микросервис для предсказания стоимости жилья в США** на основе ансамблевых моделей (`StackingRegressor`, `RandomForest`, `LightGBM`).  
Реализован REST API на Flask и контейнеризирован с помощью Docker.  

- 📁 **Папка проекта:** `housing-price-prediction`  
- ⚙️ **Состав:** `app/`, `artifacts/`, `src/`, `Dockerfile`, `requirements.txt`, `README.md`  
- 🧠 **Навыки:** Flask API, Docker, ML-модели, Stacking, LightGBM, RandomForest, предобработка данных, деплой моделей  

---

### 👥 [Segmentation_Clients](./Segmentation-clients)

📊 **Проект по сегментации клиентов онлайн-магазина подарков** с использованием RFM-анализа и методов кластеризации.  

- 📁 **Папка проекта:** `Segmentation-clients`  
- ⚙️ **Состав:** ноутбук, папка `data`, `README.md` с описанием проекта  
- 🧠 **Навыки:** анализ данных, RFM-анализ, кластеризация, визуализация, Python, pandas, seaborn  


---

### 🏦 [Bank_Deposit_Prediction](./bank-deposit-prediction)

📈 **Проект по предсказанию подписки клиента на банковский депозит** по данным маркетинговых кампаний.  
Решается задача бинарной классификации: определение вероятности открытия депозита после маркетингового звонка.

- 📁 **Папка проекта:** `bank_deposit_prediction`
- ⚙️ **Состав:**
  - `Project_4_ML.ipynb` — полный ML-pipeline: EDA → обработка → моделирование → оценка качества  
  - `README.md` — описание проекта  
  - `requirements.txt` — зависимости  
  - `data/` — локальная папка с датасетом (в `.gitignore`)
- 🧠 **Навыки:**  
  - анализ данных и визуализации  
  - кодирование категориальных признаков  
  - работа с дисбалансом  
  - модели классификации: Logistic Regression, Decision Tree, RandomForest, GradientBoosting  
  - подбор гиперпараметров  
  - оценка качества (F1, Accuracy)

- 📊 **Результаты:**  
  - *F1-score (train):* **0.92**  
  - *F1-score (test):* **0.82**  
  - *Accuracy (test):* **0.83**  
  Модель успешно определяет клиентов с высокой вероятностью оформления депозита и может использоваться для приоритизации звонков в маркетинговых кампаниях.

---

### 📊 [Credit Scoring — ML Pipeline + FastAPI Service](./Credit-scoring)

💳 **Микросервис для скоринга клиентов банка и предсказания вероятности дефолта.**  
Проект включает полный ML-pipeline: от предобработки данных и обучения модели LightGBM до деплоя FastAPI-сервиса в Docker.

- 📁 **Папка проекта:** `Credit-scoring`  
- ⚙️ **Состав:**
  - `notebook.ipynb` — полный ML-pipeline: EDA → feature engineering → LightGBM → подбор порога → оценка качества  
  - `app/service.py` — FastAPI-микросервис для онлайн-скоринга  
  - `model.pkl` — обученная модель + пайплайн предобработки  
  - `Dockerfile`, `docker-compose.yml` — инфраструктура для деплоя  
  - `requirements-deploy.txt` — зависимости для контейнера  
  - `README.md` — описание проекта  

- 🧠 **Навыки:**
  - анализ данных и фичеинжиниринг  
  - создание инженерных признаков (логарифмы, нормализации, отношения)  
  - LightGBM + sklearn Pipeline  
  - подбор оптимального порога (критерий Юдена)  
  - создание REST API на FastAPI  
  - работа с Docker и микросервисной архитектурой  

- 📊 **Результаты:**
  - *ROC-AUC:* **0.73**  
  - *Оптимальный порог выдачи кредита:* **0.1054**  
  - модель возвращает вероятность дефолта и решение об одобрении заявки  

- 🧪 Пример ответа сервиса:
```json
{
  "default_proba": 0.1264,
  "approved": true
}
```

---

### 🚕 [NY_Taxi_Trip_Duration](./NY-Taxi-Trip-Duration)

🗽 **Проект по прогнозированию длительности поездок такси в Нью-Йорке.**  
Задача — предсказать время поездки на основе данных о точках отправления и прибытия, времени, погодных условий и других факторов.  

- 📁 **Папка проекта:** `NY-Taxi-Trip-Duration`  
- ⚙️ **Состав:** Jupyter Notebook, `requirements.txt`, `README.md`  
- 🧠 **Навыки:** анализ данных, визуализация, работа с геолокацией (`geopy`, `haversine`), фичеинжиниринг, градиентные бустинги (`XGBoost`, `LightGBM`)  
- 📊 **Результаты:** модель демонстрирует хорошее качество прогноза времени поездки, позволяя оценивать транспортную эффективность и предсказывать загруженность маршрутов  

---

### 🔐 [Captcha-CNN-Kaggle](./captcha-cnn-kaggle)

🧠 **Проект по распознаванию символов captcha на соревновании Kaggle (SF CAPTCHA Recognition).**  
Построена сверточная нейронная сеть (CNN) с аугментацией изображений, достигшая **0.85070 Public Score** на тестовом наборе Kaggle.

- 📁 **Папка проекта:** `captcha-cnn-kaggle`  
- ⚙️ **Состав:**
  - `notebooks/` — основной Jupyter Notebook с полным ML-пайплайном  
  - `data/` — локальная папка для датасета (в `.gitignore`)  
  - `README.md` — описание проекта  
  - `requirements.txt` — зависимости
- 🧠 **Навыки:**  
  - свёрточные нейронные сети (CNN)  
  - TensorFlow / Keras  
  - аугментация изображений  
  - нормализация и предобработка  
  - визуализация метрик обучения  
  - формирование `submission.csv` для Kaggle  
- 📊 **Результаты:**  
  - *Train Accuracy:* ≈ 0.75  
  - *Validation Accuracy:* ≈ 0.84  
  - *Kaggle Public Score:* **0.85070**

---

### 🏨 [Booking_Review_Score_Prediction](./ml-booking)

📝 **Проект по предсказанию итогового рейтинга отелей на Booking.com** на основе текстовых отзывов и дополнительных признаков.  
Включает NLP-предобработку, векторизацию TF-IDF, визуализацию данных и сравнение ML-регрессоров.

- 📁 **Папка проекта:** `ml-booking`
- ⚙️ **Состав:**
  - `ml-booking.ipynb` — полный ML-pipeline: EDA → NLP → TF-IDF → моделирование  
  - `images/` — визуализации  
  - `other_files/` — доп. файлы (стоп-слова и др.)  
  - `results/` — обученные модели/метрики (в `.gitignore`)  
  - `README.md` — подробное описание  
  - `requirements.txt` — зависимости проекта  
- 🧠 **Навыки:**  
  - NLP: очистка текста, лемматизация, stopwords  
  - TF-IDF векторизация  
  - регрессия: RandomForest, GradientBoosting, Linear Models  
  - визуализация распределений и метрик  
  - анализ важности признаков  
- 📊 **Результаты:**  
  - *MAE:* ~0.32  
  - *RMSE:* ~0.48  
  - *R²:* ~0.77  
  - модель стабильно предсказывает итоговый рейтинг отеля с хорошей точностью.

---

### 📰 [Articles_Sharing_and_Reading](./Articles-sharing-and-reading)

📚 **Два рекомендательных алгоритма для платформы обмена и чтения статей.**  
В проекте реализованы **персональные рекомендации (SVD)** и **популярностная модель**, позволяющие сравнить подходы и оценить качество рекомендаций.

- 📁 **Папка проекта:** `Articles-sharing-and-reading`
- ⚙️ **Состав:**
  - `personalized_recommender.ipynb` — модель персональных рекомендаций (Surprise SVD)  
  - `popularity_recommender.ipynb` — baseline модель рекомендаций по популярности  
  - `data/` — локальные данные (исключены через `.gitignore`)  
  - `surprise-env/` — отдельное окружение для библиотеки Surprise (в `.gitignore`)  
  - `README.md` — описание проекта  
- 🧠 **Навыки:**  
  - рекомендательные системы (CF, SVD)  
  - анализ взаимодействий пользователей  
  - популярностные baseline-модели  
  - метрики качества рекомендаций: RMSE, Precision@k, Recall@k  
  - обработка данных: pandas, numpy  
- 📊 **Результаты:**  
  - *Personal SVD RMSE:* ~0.82  
  - *Precision@k / Recall@k* значительно выше baseline  
  - популярностная модель эффективна в сценариях cold-start  
  - персональная модель лучше улавливает индивидуальные предпочтения пользователей и дает более релевантные рекомендации.


## 🛠 Как использовать проекты

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/sudecii/Projects.git

📬 Контакты
	•	💼 GitHub: @sudecii￼
	•	💬 Telegram: @yokyokyokyokyoky￼
	•	📫 Всегда открыт для общения, обмена опытом и обсуждения проектов!