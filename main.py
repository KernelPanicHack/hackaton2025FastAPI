from collections import Counter
from typing import List, Dict, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "models/categorizing/sgd_classifier_model.pkl"
VECTORIZER_PATH = "models/categorizing/tfidf_vectorizer.pkl"

app = FastAPI(
    title="API моделей категоризации и прогнозирования",
    description="API для категоризации списка товаров и прогнозирования трат пользователя",
    version="1.0.0"
)

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Модели успешно загружены из {MODEL_PATH} и {VECTORIZER_PATH}")
except Exception as e:
    print(f"Ошибка загрузки моделей: {str(e)}")
    model = None
    vectorizer = None

CONFIDENCE_THRESHOLD = 0.5


# Модели для категоризации товаров
class CategoryItemsRequest(BaseModel):
    items: List[str]


class CategoryResponse(BaseModel):
    category: str
    categories: Optional[Dict[str, List[int]]] = None
    status: str
    confidence: Optional[float] = None


# Модели для прогнозирования трат
class ExpensePredictionRequest(BaseModel):
    user_id: str
    period: Optional[int] = 30


class ExpensePredictionResponse(BaseModel):
    predictions: Dict[str, float]
    status: str


@app.post("/api/categorize-items", response_model=CategoryResponse)
async def categorize_items(request: CategoryItemsRequest):
    """
    Эндпоинт для категоризации списка товаров по их названиям
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Модели не загружены")

    try:

        items = request.items
        vect_items = vectorizer.transform(items)

        probabilities = model.predict_proba(vect_items)

        predictions = []

        for i in range(len(probabilities)):
            max_prob = np.max(probabilities[i])

            if max_prob < CONFIDENCE_THRESHOLD:
                predictions.append('Misc')
            else:
                predicted_class = model.classes_[np.argmax(probabilities[i])]
                predictions.append(predicted_class)

        category_counts = Counter(predictions)
        most_common_category = category_counts.most_common(1)[0][0]
        confidence = category_counts[most_common_category] / len(predictions)

        return {
            "category": most_common_category,
            "status": "success",
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при категоризации товаров: {str(e)}")


@app.post("/api/predict-expenses", response_model=ExpensePredictionResponse)
async def predict_expenses(request: ExpensePredictionRequest):
    """
    Эндпоинт для прогнозирования трат пользователя
    """
    return {"predictions": {}, "status": "not implemented yet"}


# Health-check эндпоинт
@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "api_version": "1.0.0",
        "models_loaded": {
            "categorizing": model is not None and vectorizer is not None
        }
    }
