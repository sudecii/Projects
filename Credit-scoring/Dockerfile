FROM python:3.12-slim

# 1. Доставляем системную библиотеку для LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app 

COPY app/requirements-deploy.txt ./requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ . 

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]


