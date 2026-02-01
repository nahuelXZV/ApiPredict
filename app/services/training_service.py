import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  
DATA_PATH = BASE_DIR / "storage" / "data" / "ventas_olimpia.csv"
MODEL_PATH = BASE_DIR / "storage" / "models" / "modelo_knn_ol.pkl"

class TrainingService:
    def train_from_excel(self) -> None:
        # 1. Cargar dataset
        df = pd.read_csv(DATA_PATH)
        print("Dataset original")
        print(df.head())
        df.info()

        # 2. Preprocesamiento fecha
        df["fecha"] = pd.to_datetime(df["fecha"])
        df["anio"] = df["fecha"].dt.year
        df["mes"] = df["fecha"].dt.month
        df["dia_semana"] = df["fecha"].dt.dayofweek

        # 3. Filtrado de columnas
        df_filter = df.drop(
            columns=[
                "nombre_empresa",
                "codigo_producto",
                "nombre_producto",
                "precio_unitario",
                "descuento_unitario",
                "precio_total"
            ]
        )

        # 4. Features
        cat_features = [
            "codigo_empresa",
            "zona",
            "tipo_comerciante",
            "categoria_producto",
            "subcategoria_producto",
            "temporada"
        ]

        num_features = [
            "anio",
            "mes",
            "dia_semana",
            "cantidad"
        ]

        # 5. Preprocesamiento
        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
                ("num", StandardScaler(), num_features)
            ]
        )

        # 6. Modelo
        model = Pipeline([
            ("prep", preprocess),
            ("knn", NearestNeighbors(
                n_neighbors=20,
                metric="cosine"
            ))
        ])

        # 7. Entrenamiento
        model.fit(df_filter)

        # 8. Guardar modelo
        joblib.dump(
            {
                "model": model,
                "df": df
            },
            MODEL_PATH
        )

        print(f"Modelo guardado en {MODEL_PATH}")
