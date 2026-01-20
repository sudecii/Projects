# 🧩 Segmentation of Online Store Clients

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-3793EF.svg?logo=plotly&logoColor=white)](https://seaborn.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 🧠 Описание проекта

Проект посвящён **сегментации клиентов онлайн-магазина подарков** на основе **RFM-анализа** и **кластеризации (K-Means)**.  
Цель — выделить ключевые группы покупателей для персонализированных маркетинговых стратегий и повышения лояльности.

---

## 📊 Этапы проекта

1. **Загрузка и предобработка данных**
   - Очистка от дубликатов и возвратов  
   - Удаление заказов с нулевой стоимостью  
   - Работа с пропусками  

2. **Разведывательный анализ данных (EDA)**
   - Распределение заказов по странам и месяцам  
   - Анализ активности клиентов и повторных покупок  

3. **Построение RFM-таблицы**
   - Метрики Recency, Frequency, Monetary  
   - Нормализация и распределение значений  

4. **Кластеризация клиентов**
   - Определение оптимального числа кластеров методом “локтя”  
   - Применение K-Means  
   - Визуализация результатов кластеризации  

5. **Интерпретация сегментов и рекомендации**
   - Определение групп:  
     - 🟢 **Лояльные клиенты** — совершают покупки часто и на большие суммы  
     - 🟡 **Перспективные** — проявляют интерес, но нерегулярно  
     - 🔴 **Спящие клиенты** — давно неактивны, требуют повторного вовлечения  

---



---

## ⚙️ Используемые технологии

- **Python:** pandas, numpy, matplotlib, seaborn, scikit-learn  
- **Методы:** RFM-анализ, K-Means, нормализация данных, визуализация кластеров  
- **Среда разработки:** Jupyter Notebook  

---

## 📂 Структура проекта

```text
Segmentation_clients/
│
├── data/                                   # Исходные данные
├── PROJECT-6._Сегментация_клиентов_онлайн-магазина.ipynb  # Основной ноутбук с кодом и анализом
└── README.md                               # Описание проекта
```

---

## 🏷 Теги

`machine-learning` `data-analysis` `clustering` `rfm-analysis` `customer-segmentation` `python`

---

