# 🏨 Booking Review Score Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/NLP-Text%20Processing-green.svg)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)]()

---

## 🎯 Цель проекта

Построить модель, которая **предсказывает итоговый рейтинг отеля** на Booking.com на основе текстов отзывов и дополнительных признаков.  
Проект объединяет **EDA, NLP, машинное обучение и визуализацию данных**.

---

## 📌 Основные задачи

- Загрузка и первичная обработка датасета  
- EDA и выявление закономерностей  
- NLP-обработка отзывов: очистка, лемматизация, стоп-слова  
- Векторизация текстов с помощью **TF-IDF**  
- Обучение нескольких моделей регрессии  
- Тюнинг гиперпараметров  
- Оценка качества и анализ важности признаков  

---

## 📊 Используемые модели

- Linear Regression  
- Ridge / Lasso  
- RandomForestRegressor 🌲  
- GradientBoostingRegressor  

---

## 📈 Результаты

Лучшая модель: **RandomForestRegressor**

Достигнутые метрики:

| Метрика | Значение |
|--------|----------|
| **MAE** | ~0.32 |
| **RMSE** | ~0.48 |
| **R²** | ~0.77 |

**Ключевые признаки:**
- sentiment-метрики  
- длина текста  
- TF-IDF важные слова  
- страна рецензента  
- оценки гостей по отдельным аспектам  

---

## 🗂 Структура проекта
ml-booking/
├── data/                ← датасет (ignored в .gitignore)
│   └── reviews.csv
│
├── images/              ← визуализации и графики
│   ├── corr_matrix.png
│   ├── feature_importance.png
│   └── score_distribution.png
│
├── other_files/         ← вспомогательные материалы
│   ├── stopwords.txt
│   └── text_preprocessing_notes.md
│
├── results/             ← сохранённые модели и метрики (ignored в .gitignore)
│   ├── rf_model.pkl
│   ├── gridsearch_results.json
│   └── predictions_sample.csv
│
├── ml-booking.ipynb     ← основной ноутбук с исследованием и моделированием
└── requirements.txt     ← список зависимостей

## 🛠 Установка и запуск

```bash
git clone https://github.com/sudecii/Projects.git
cd Projects/ml-booking

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

📚 Технологии
	•	Python 3.10+
	•	pandas / numpy
	•	scikit-learn
	•	matplotlib / seaborn
	•	nltk, TF-IDF