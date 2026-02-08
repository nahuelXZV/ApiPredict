from fastapi import APIRouter
from app.services.trainingService import TrainingService

router = APIRouter()
service = TrainingService()

@router.post("/train")
def train_model():
    service.train_from_excel()
    return {
        "status": "ok",
        "message": "Modelo entrenado y guardado correctamente"
    }
