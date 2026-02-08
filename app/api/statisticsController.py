from fastapi import APIRouter, HTTPException
from app.services.statisticsService import StatisticsService

router = APIRouter()
service = StatisticsService()

@router.get("/stats/resumen", tags=["Estadísticas"])
def resumen_general():
    try:
        return service.resumen_general()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/productos", tags=["Estadísticas"])
def ventas_por_producto(top_n: int = 10):
    return service.ventas_por_producto(top_n)


@router.get("/stats/categorias", tags=["Estadísticas"])
def ventas_por_categoria():
    return service.ventas_por_categoria()


@router.get("/stats/temporadas", tags=["Estadísticas"])
def mas_vendidos_por_temporada(top_n: int = 5):
    return service.mas_vendidos_por_temporada(top_n)


@router.get("/stats/fechas", tags=["Estadísticas"])
def ventas_por_fecha():
    return service.ventas_por_fecha()

@router.get("/empresa/{codigo_empresa}", tags=["Empresa"])
def datos_empresa(codigo_empresa: str):
    return service.obtener_datos_empresa(codigo_empresa)

@router.get("/empresa/{codigo_empresa}/resumen", tags=["Empresa"])
def resumen_empresa(codigo_empresa: str):
    return service.resumen_empresa(codigo_empresa)