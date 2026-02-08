from fastapi import FastAPI
from app.api import trainController
from app.api import pedidoController
from app.api import statisticsController


def create_app() -> FastAPI:
    app = FastAPI(
        title="API Pedido Sugerido",
        version="1.0.0",
        description="API para cálculo y consulta de pedidos sugeridos"
    )

    # Routers
    app.include_router(trainController.router, prefix="/api", tags=["Training"])
    app.include_router(pedidoController.router,prefix="/api",tags=["Predicciones"])
    app.include_router(statisticsController.router,prefix="/api",tags=["Estadísticas"])
    
    # Eventos
    @app.on_event("startup")
    async def startup():
        print("Iniciando API...")

    @app.on_event("shutdown")
    async def shutdown():
        print("Apagando API...")

    return app


app = create_app()