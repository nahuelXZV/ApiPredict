from fastapi import APIRouter, HTTPException
from app.schemas.prediction_input import ClienteDTO
from app.services.prediction_service import PredictionService

router = APIRouter()
service = PredictionService()

@router.post("/predict", tags=["Predicciones"])
def predict(cliente: ClienteDTO, top_n: int = 10, apply_weights: bool = False):
    try:
        return service.predict(cliente, top_n=top_n, apply_weights=apply_weights)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
