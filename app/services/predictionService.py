import joblib
import pandas as pd
from pathlib import Path
from app.schemas.prediction_input import ClienteDTO

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "storage" / "models" / "modelo_knn_ol.pkl"
PRODUCTOS_PATH = BASE_DIR / "storage" / "data" / "datos_productos.csv"

class PredictionService:

    def predict(self, dto: ClienteDTO, top_n: int = 10, apply_weights: bool = False) -> list[dict]:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Modelo no entrenado")

        #  Cargar modelo
        obj = joblib.load(MODEL_PATH)
        model = obj["model"]
        df = obj["df"]

        #  Crear DataFrame cliente
        cliente_df = pd.DataFrame([dto.dict()])

        #  Preprocesar fecha
        cliente_df["fecha"] = pd.to_datetime(cliente_df["fecha"])
        cliente_df["anio"] = cliente_df["fecha"].dt.year
        cliente_df["mes"] = cliente_df["fecha"].dt.month
        cliente_df["dia_semana"] = cliente_df["fecha"].dt.dayofweek

        #  Vectorizar cliente
        cliente_vec = model["prep"].transform(cliente_df)
        distances, indices = model["knn"].kneighbors(cliente_vec)

        # Armar recomendaciones
        df_reco = df.iloc[indices[0]][[
            "codigo_producto",
            "codigo_empresa",
            "nombre_producto",
            "categoria_producto",
            "zona",
            "temporada",
            "tipo_comerciante",
        ]].copy()
        df_reco["distancia"] = distances[0]
        
        # Cargar pesos de producto
        df_productos = pd.read_csv(PRODUCTOS_PATH)
        df_productos["peso_manual"] = df_productos["peso_manual"].fillna(1.0)
        df_productos["peso_calculado"] = df_productos["peso_calculado"].fillna(1.0)
        df_productos["peso_final"] = ( df_productos["peso_manual"] * df_productos["peso_calculado"] )
        
        # Agregar el peso al DataFrame de recomendaciones
        df_reco = df_reco.merge(
                df_productos[["codigo_producto", "peso_final"]],
                on="codigo_producto",
                how="left"
            )
        df_reco["peso_final"] = df_reco["peso_final"].fillna(1.0)

        # Quitar productos no recomendados df_productos["no_recomendar"]
        productos_excluidos = df_productos[df_productos["no_recomendar"] == True]
        df_reco = df_reco[~df_reco["codigo_producto"].isin(productos_excluidos["codigo_producto"])]

        if not apply_weights:
           return self.procesar_respuesta(df_reco, top_n=top_n, order="distancia")

        # 7. Score final 
        df_reco["similitud"] = 1 - df_reco["distancia"]
        df_reco["score_final"] = (df_reco["similitud"] * df_reco["peso_final"])

        return self.procesar_respuesta(df_reco, top_n=top_n, order="score_final")

    def procesar_respuesta(self, df_reco: pd.DataFrame, top_n: int, order: str) -> list[dict]:
        df_reco = (
            df_reco
            .sort_values(order, ascending=False)
            .head(top_n)
        )

        columnas = [
            "codigo_producto",
            "nombre_producto",
            "categoria_producto",
            "distancia",
            "peso_final",
        ]
        if "score_final" in df_reco.columns:
            columnas.append("score_final")
        return df_reco[columnas].to_dict(orient="records")
