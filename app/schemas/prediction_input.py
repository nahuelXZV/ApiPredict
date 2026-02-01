from pydantic import BaseModel

class ClienteDTO(BaseModel):
    codigo_empresa: str
    zona: str
    tipo_comerciante: str
    fecha: str
    categoria_producto: str
    subcategoria_producto: str | None = None
    cantidad: int
    temporada: str
