# 1) Лёгкая база с Python
FROM python:3.10-slim

# 2) Базовые настройки Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3) Рабочая директория внутри контейнера
WORKDIR /app

# 4) Библиотека OpenMP для lightgbm/xgboost 
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# 5) Ставим зависимости слоями 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6) Копируем только то, что нужно для инференса
COPY app/ app/
COPY artifacts/ artifacts/

# 7) Порт сервиса
EXPOSE 8000

# 8) Production-сервер — gunicorn (импортирует app из app/server.py)
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app.server:app"]