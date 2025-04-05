from typing import Dict
import logging
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API",
    description="API для сервисов",
    version="1.0.0"
)

try:
    model = joblib.load('models/categorizing/xgboost_model.pkl')
    label_encoder = joblib.load('models/categorizing/label_encoder.pkl')
    logger.info("Модель и энкодер успешно загружены")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {str(e)}")
    model = None
    label_encoder = None

class TransactionRequest(BaseModel):
    Date: str
    WithDrawal: float
    Balance: float

class TransactionResponse(BaseModel):
    category: str
    status: str

@app.get("/")
async def root():
    return {"message": "API готов к работе"}

@app.post("/api/categorize-transaction", response_model=TransactionResponse)
async def categorize_transaction(request: TransactionRequest):
    """
    Эндпоинт для классификации транзакции на основе даты, суммы снятия и баланса
    """
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
        
    try:
        logger.info(f"Получен запрос: {request.dict()}")
        
        date_obj = pd.to_datetime(request.Date)
        
        features = {
            'Year': date_obj.year,
            'Month': date_obj.month,
            'Day': date_obj.day,
            'Withdrawal': request.WithDrawal,
            'Deposit': 0,
            'Balance': request.Balance
        }
        
        X = pd.DataFrame([features])
        logger.info(f"Подготовленные данные: {X}")
        
        prediction = model.predict(X)[0]
        
        if hasattr(label_encoder, 'inverse_transform'):
            category = label_encoder.inverse_transform([prediction])[0]
        else:
            category = str(prediction)
            
        logger.info(f"Предсказанная категория: {category}")
        
        return {
            "category": category,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Ошибка при классификации транзакции: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при классификации транзакции: {str(e)}")
