# 🚀 ML Projects Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-ML%20Service-009688.svg?logo=fastapi&logoColor=white)]()
[![Flask](https://img.shields.io/badge/Flask-API-lightgrey.svg?logo=flask&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED.svg?logo=docker&logoColor=white)]()
[![Machine Learning](https://img.shields.io/badge/ML-Models-orange.svg?logo=scikit-learn&logoColor=white)]()
[![Data Science](https://img.shields.io/badge/Data%20Science-EDA%20%7C%20Features-yellow.svg?logo=pandas&logoColor=white)]()

---

# 👨‍💻 About Me

Я начинающий Data Scientist, который развивает навыки в области анализа данных, машинного обучения и построения end-to-end ML-решений.  
В проектах делаю упор на:  
- качественный EDA и предобработку данных,  
- построение и сравнение ML-моделей,  
- разработку сервисов для продакшн-использования,  
- оформление решений с учётом бизнес-контекста.  

Открыт к обучению, командной работе и участию в реальных проектах.

---

# 🧩 Основные проекты

---

## 🏡 Housing Price Prediction — ML Microservice  
**TL;DR:** Микросервис на Flask + Docker, предсказывающий стоимость недвижимости с помощью ансамблей (LightGBM, RF, Stacking).  

### 🎯 Бизнес-задача  
Оценка стоимости объектов важна для:  
- банков (ипотечная оценка),  
- страховых компаний,  
- риелторских сервисов.  

**Ценность:** автоматизация оценки и ускорение процессов.

### 🧠 Стек & навыки  
Flask API, Docker, StackingRegressor, EDA, feature engineering.

📁 Папка: [housing-price-prediction](./housing-price-prediction)

---

## 👥 Customer Segmentation (RFM + Clustering)  
**TL;DR:** RFM-анализ и кластеризация клиентов для персонализации маркетинга.

### 🎯 Бизнес-задача  
Компания стремится повысить удержание и доходность клиентов:  
- определить ключевые сегменты,  
- выделить VIP-клиентов,  
- понять группы, требующие внимания.  

**Ценность:** оптимизация маркетинговых решений.

### 🧠 Стек & навыки  
RFM, KMeans, визуализация, pandas, seaborn.

📁 Папка: [Segmentation-clients](./Segmentation-clients)

---

## 🏦 Bank Deposit Prediction  
**TL;DR:** Модель классификации для прогнозирования отклика клиента на депозитное предложение.  

### 🎯 Бизнес-задача  
Повышение эффективности маркетинговых звонков:  
- определение клиентов, с большей вероятностью готовых открыть депозит,  
- сокращение затрат на массовый обзвон.  

**Ценность:** рост конверсии и экономия ресурсов.

### 📊 Метрики  
- F1 (test): **0.82**  
- Accuracy (test): **0.83**

### 🧠 Стек & навыки  
Работа с дисбалансом, One-Hot Encoding, Boosting, F1-оптимизация.

📁 Папка: [bank-deposit-prediction](./bank-deposit-prediction)

---

## 💳 Credit Scoring — ML Pipeline + FastAPI Service  
**TL;DR:** Полноценный ML-pipeline + FastAPI-сервис для расчёта вероятности дефолта.  

### 🎯 Бизнес-задача  
Автоматизация принятия решений и оценка рисков:  
- скоринг заявок,  
- выделение клиентов высокого риска,  
- ускорение рассмотрения.  

**Ценность:** стандартизация и повышение качества решений.

### 📊 Метрики  
- ROC-AUC: **0.73**  
- Оптимальный порог: **0.1054**

### 🧠 Стек & навыки  
LightGBM, Feature engineering, Pipeline, FastAPI, Docker.

📁 Папка: [Credit-scoring](./Credit-scoring)

---

## 🚕 NY Taxi Trip Duration  
**TL;DR:** Прогноз времени поездки такси по координатам, времени и погодным данным.

### 🎯 Бизнес-задача  
Актуально для:  
- предсказания ETA,  
- динамического ценообразования,  
- оптимизации логистики.  

**Ценность:** повышение точности прогнозов и удобства пользователей.

### 🧠 Стек & навыки  
Haversine distance, geopy, feature engineering, XGBoost, LightGBM.

📁 Папка: [NY-Taxi-Trip-Duration](./NY-Taxi-Trip-Duration)

---

# 📎 Дополнительные проекты  
_Показывают широту знаний и эксперименты в разных ML-направлениях._

---

## 🔐 Captcha Recognition — CNN (Kaggle)  
**TL;DR:** CNN-модель для распознавания CAPTCHA. Kaggle Public Score: **0.85070**.

### 🎯 Бизнес-ценность  
Автоматизация проверки пользователей и борьба с ботами.

### 🧠 Навыки  
TensorFlow/Keras, аугментация, сверточные сети.

📁 Папка: [captcha-cnn-kaggle](./captcha-cnn-kaggle)

---

## 📰 Article Recommendation System  
**TL;DR:** Персональные рекомендации на SVD + популярностная модель.

### 🎯 Бизнес-ценность  
Рост CTR, увеличение времени чтения, снижение churn.

### 🧠 Навыки  
SVD, CF, RMSE, Precision@k, Recall@k.

📁 Папка: [Articles-sharing-and-reading](./Articles-sharing-and-reading)

---

# 🛠 Как запустить проекты

```bash
git clone https://github.com/sudecii/Projects.git

📬 Контакты
	•	💬 Telegram: @yokyokyokyokyoky


Всегда открыт для общения и сотрудничества!