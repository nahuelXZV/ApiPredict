import joblib
import pandas as pd
from pathlib import Path
from app.schemas.prediction_input import ClienteDTO

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "storage" / "models" / "modelo_knn_ol.pkl"
PESOS_PATH = BASE_DIR / "storage" / "data" / "pesos_productos.csv"

class PredictionService:

    def predict(self, dto: ClienteDTO, top_n: int = 10, apply_weights: bool = False) -> list[dict]:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Modelo no entrenado")

        # 1. Cargar modelo
        obj = joblib.load(MODEL_PATH)
        model = obj["model"]
        df = obj["df"]

        # 2. Crear DataFrame cliente
        cliente_df = pd.DataFrame([dto.dict()])

        # 3. Preprocesar fecha
        cliente_df["fecha"] = pd.to_datetime(cliente_df["fecha"])
        cliente_df["anio"] = cliente_df["fecha"].dt.year
        cliente_df["mes"] = cliente_df["fecha"].dt.month
        cliente_df["dia_semana"] = cliente_df["fecha"].dt.dayofweek

        # 4. Vectorizar cliente
        cliente_vec = model["prep"].transform(cliente_df)
        distances, indices = model["knn"].kneighbors(cliente_vec)

        # 5. Armar recomendaciones
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
        
        # 6. Cargar pesos de producto
        pesos_df = pd.read_csv(PESOS_PATH)
        pesos_df["peso_manual"] = pesos_df["peso_manual"].fillna(1.0)
        pesos_df["peso_calculado"] = pesos_df["peso_calculado"].fillna(1.0)
        pesos_df["peso_final"] = ( pesos_df["peso_manual"] * pesos_df["peso_calculado"] )
        
        # 7. Enriquecer recomendaciones con pesos
        df_reco = df_reco.merge(
                pesos_df[["codigo_producto", "peso_final"]],
                on="codigo_producto",
                how="left"
            )
        df_reco["peso_final"] = df_reco["peso_final"].fillna(1.0)

        if not apply_weights:
           return df_reco[[
                "codigo_producto",
                "nombre_producto",
                "categoria_producto",
                "distancia",
                "peso_final"
            ]].to_dict(orient="records")

        # 7. Score final 
        df_reco["similitud"] = 1 - df_reco["distancia"]

        df_reco["score_final"] = (
            df_reco["similitud"] * df_reco["peso_final"]
        )

        # 8. Ranking final
        df_reco = (
            df_reco
            .sort_values("score_final", ascending=False)
            .head(top_n)
        )

        return df_reco[[
            "codigo_producto",
            "nombre_producto",
            "categoria_producto",
            "distancia",
            "peso_final",
            "score_final"
        ]].to_dict(orient="records")
