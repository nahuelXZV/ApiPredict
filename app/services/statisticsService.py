import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "storage" / "data" / "ventas_olimpia.csv"

class StatisticsService:

    def __init__(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError("Archivo de ventas no encontrado")

        self.df = pd.read_csv(DATA_PATH)
        self.df["fecha"] = pd.to_datetime(self.df["fecha"])

    # KPIs generales
    def resumen_general(self) -> dict:
        return {
            "total_registros": len(self.df),
            "total_productos_distintos": self.df["codigo_producto"].nunique(),
            "total_categorias": self.df["categoria_producto"].nunique(),
            "total_zonas": self.df["zona"].nunique(),
            "total_tipos_comerciantes": self.df["tipo_comerciante"].nunique(),
        }

    # Ventas por producto
    def ventas_por_producto(self, top_n: int = 10) -> list[dict]:
        df = (
            self.df
            .groupby(["codigo_producto", "nombre_producto"], as_index=False)
            .agg({
                "cantidad": "sum",
                "precio_total": "sum"
            })
            .sort_values("cantidad", ascending=False)
            .head(top_n)
        )

        return df.to_dict(orient="records")

    # Ventas por categoría
    def ventas_por_categoria(self) -> list[dict]:
        df = (
            self.df
            .groupby("categoria_producto", as_index=False)
            .agg({
                "cantidad": "sum",
                "precio_total": "sum"
            })
            .sort_values("cantidad", ascending=False)
        )

        return df.to_dict(orient="records")

    # Más vendidos por temporada
    def mas_vendidos_por_temporada(self, top_n: int = 5) -> list[dict]:
        df = (
            self.df
            .groupby(["temporada", "codigo_producto", "nombre_producto"], as_index=False)
            .agg({"cantidad": "sum"})
            .sort_values(["temporada", "cantidad"], ascending=[True, False])
        )

        return (
            df
            .groupby("temporada")
            .head(top_n)
            .to_dict(orient="records")
        )

    # Ventas por fecha
    def ventas_por_fecha(self) -> list[dict]:
        df = (
            self.df
            .groupby(self.df["fecha"].dt.date, as_index=False)
            .agg({
                "cantidad": "sum",
                "precio_total": "sum"
            })
            .rename(columns={"fecha": "fecha"})
        )

        return df.to_dict(orient="records")

    def obtener_datos_empresa(self, codigo_empresa: str) -> list[dict]:
        self.df["codigo_empresa"] = self.df["codigo_empresa"].astype(str)
        df_empresa = self.df[self.df["codigo_empresa"] == codigo_empresa]
        if df_empresa.empty:
            return []

        return df_empresa.to_dict(orient="records")

    def resumen_empresa(self, codigo_empresa: str) -> dict:
        self.df["codigo_empresa"] = self.df["codigo_empresa"].astype(str)
        df_empresa = self.df[self.df["codigo_empresa"] == codigo_empresa]

        if df_empresa.empty:
            return {}

        return {
            "codigo_empresa": codigo_empresa,
            "total_registros": len(df_empresa),
            "productos_distintos": df_empresa["codigo_producto"].nunique(),
            "cantidad_total": int(df_empresa["cantidad"].sum()),
            "ventas_totales": float(df_empresa["precio_total"].sum()),
            "categorias": df_empresa["categoria_producto"].nunique(),
            "temporadas": df_empresa["temporada"].nunique()
        }
