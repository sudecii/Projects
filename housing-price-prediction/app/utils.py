import json
from typing import List, Dict, Any
import pandas as pd


def load_feature_list(path: str) -> List[str]:
    """Читает список признаков из JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_input(payload: Dict[str, Any], feature_list: List[str]) -> pd.DataFrame:
    """
    Приводит один объект (словарь признаков) к DataFrame в нужном порядке.
    Отсутствующие признаки заполняются None — дальнейшая обработка/импьютация внутри пайплайна.
    """
    row = {col: payload.get(col, None) for col in feature_list}
    return pd.DataFrame([row], columns=feature_list)


def prepare_batch(payload_list: List[Dict[str, Any]], feature_list: List[str]) -> pd.DataFrame:
    """
    Приводит список объектов (каждый — словарь признаков) к DataFrame в нужном порядке.
    Отсутствующие признаки заполняются None.
    """
    rows = [{col: item.get(col, None) for col in feature_list} for item in payload_list]
    return pd.DataFrame(rows, columns=feature_list)