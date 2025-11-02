# Импорт нужных библиотек
import os
import json
import logging
from typing import List, Dict, Any
import sys

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
)
log = logging.getLogger("server")

# чтобы импортировать utils 
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# наши утилиты лежат в этой же папке app/
from app.utils import load_feature_list, prepare_input, prepare_batch

# пути к артефактам 
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "stacking_model.joblib")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_list.json")  

# загрузка модели и списка признаков 
try:
    model = joblib.load(MODEL_PATH)
    feature_list: List[str] = load_feature_list(FEATURES_PATH)
    log.info("Модель и список признаков загружены. n_features=%d", len(feature_list))
except Exception:
    log.exception("Не удалось загрузить модель/признаки")
    raise

# Создаем объект приложения
app = Flask(__name__)

# Состояние сервера
@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"}), 200

# Список признаков
@app.get("/features")
def features() -> Any:
    return jsonify({"features": feature_list, "count": len(feature_list)}), 200

# Предикт одного объекта
@app.post("/predict")
def predict_single() -> Any:
    """
    Ожидает JSON-объект с признаками одного дома.
    Возвращает предсказанную цену в USD.
    """
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    if not isinstance(payload, dict) or not payload:
        return jsonify({"error": "expected JSON object with features"}), 400

    try:
        X = prepare_input(payload, feature_list)   # DataFrame в нужном порядке колонок
        y_pred = model.predict(X)                  # модель — это TransformedTargetRegressor => выдаёт USD
        price = round(float(np.asarray(y_pred).ravel()[0]), 2)
        return jsonify({"prediction_usd": price}), 200
    except Exception as e:
        log.exception("Ошибка предсказания (single): %s", e)
        return jsonify({"error": "prediction failed"}), 500

# Предикт нескольких объектов
@app.post("/predict_batch")
def predict_batch_ep() -> Any:
    """
    Ожидает JSON-массив объектов (список домов).
    Возвращает список цен в USD по порядку.
    """
    payload_list = request.get_json(silent=True)
    if not isinstance(payload_list, list) or not payload_list:
        return jsonify({"error": "expected JSON array of objects"}), 400

    try:
        X = prepare_batch(payload_list, feature_list)
        y_pred = model.predict(X)
        prices = [round(float(v), 2) for v in np.asarray(y_pred).ravel().tolist()]
        return jsonify({"predictions_usd": prices, "count": len(prices)}), 200
    except Exception as e:
        log.exception("Ошибка предсказания (batch): %s", e)
        return jsonify({"error": "batch prediction failed"}), 500

#  ЗАПУСК 
if __name__ == "__main__":
    # запуск из корня проекта:  python3 app/server.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)