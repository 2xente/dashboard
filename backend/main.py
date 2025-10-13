from fastapi import FastAPI, HTTPException
import numpy as np
import logging
import pandas as pd
from backend import methode 
from pandas.api.types import is_numeric_dtype
import shap
from typing import List
from sklearn.pipeline import Pipeline
import math


def json_safe(obj):
    """
    Rend 'obj' JSON-sérialisable :
    - remplace NaN / ±Inf par None
    - convertit les types numpy/pandas vers des types Python natifs
    - nettoie récursivement listes/dicts
    """
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, (np.floating,)):
        v = float(obj);  return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]

    if isinstance(obj, pd.DataFrame):
        return [json_safe(rec) for rec in obj.to_dict(orient="records")]
    if isinstance(obj, (pd.Series, pd.Index)):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]

    try:
        return json_safe(obj.__dict__)
    except Exception:
        return str(obj)


logger = logging.getLogger("api")
logger.setLevel(logging.INFO)


## Charger les données de test 
#df=pd.read_csv("s3://bixentep7/data_test_with_index.csv")
df= methode.load_data()
print(df.head())

## Charger les données de test 
df_description =pd.read_csv("s3://bixentep7/HomeCredit_columns_description.csv", encoding='latin1')


## Charger le model
PKL_PATH = "s3://bixentep7/pipeline_app.pkl"
modele = pd.read_pickle(PKL_PATH) 

FEATURE_ORDER: List[str] = modele.get("feature_order", None)
ID_COLUMN = "SK_ID_CURR"
print(FEATURE_ORDER)


app = FastAPI()

def _split_pipeline(model):
    if isinstance(model, Pipeline):
        if len(model.steps) == 1:
            return None, model.steps[-1][1]
        pre = Pipeline(model.steps[:-1]); est = model.steps[-1][1]
        return pre, est
    return None, model

def _get_feature_names_after_preproc(preprocessor, raw_feature_names: List[str]) -> List[str]:
    if preprocessor is None: return raw_feature_names
    if hasattr(preprocessor, "get_feature_names_out"):
        try: return list(preprocessor.get_feature_names_out(raw_feature_names))
        except Exception: pass
    return raw_feature_names

def _map_transformed_to_raw(transformed_names: List[str], raw_feature_names: List[str]) -> List[str]:
    """Mappe chaque feature transformée vers un nom 'brut' en prenant
    le plus long nom brut qui est préfixe de la feature transformée.
    (simple et robuste pour OneHot: 'COL_A_cat' → 'COL_A')"""
    mapped = []
    for t in transformed_names:
        candidates = [r for r in raw_feature_names if t.startswith(r)]
        mapped.append(max(candidates, key=len) if candidates else t)
    return mapped



@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon API FastAPI 🚀"}

@app.get("/load_data")
def load_data():
    df=pd.read_csv("s3://bixentep7/data_test_app.csv")
    return {"data": df.to_dict(orient="records")}

@app.get("/load_model")
def load_model():
    payload = pd.read_pickle(PKL_PATH)
    print(payload.get("feature_order", None))
    return {"model": "ok"}

# a test 
@app.get("/prediction")
def prediction(sk_id_curr = str):
    """
    Fonction pour faire des prédictions sur une liste d'identifiants clients 
    """
    try:
        data_test = df[df["SK_ID_CURR"] == int(sk_id_curr)].drop("SK_ID_CURR", axis=1)
        model = modele["pipeline"]
        feature_order = modele.get("feature_order", None)
        if feature_order is not None:
            data_test = data_test[feature_order]
        result = model.predict(data_test)
        proba = model.predict_proba(data_test)[:, 1]
        return {"prediction": int(result[0]), "probability": float(proba[0])}
    except Exception as e:
        return {"error": str(e)}

@app.get("/customer_info")
def customer_info(sk_id_curr = str):
    """
    Fonction pour obtenir les informations d'un client spécifique
    """
    try:
        customer_data = df[df["SK_ID_CURR"] == int(sk_id_curr)]
        if not customer_data.empty:
            return {"customer_info": customer_data.to_dict(orient="records")[0]}
        else:
            return {"error": "Client not found"}
    except Exception as e:
        return {"error": str(e)}


#charger le df d'explication des colonne et aller chercher l'explication pour une colonne spécifique
@app.get("/columns_description")
def columns_info(column_name: str):
    """
    Fonction pour obtenir les informations d'une colonne spécifique
    """
    try:
        if column_name in df_description["Row"].dropna().tolist():
            print (f"column name {column_name}")    
            description = df_description.loc[df_description["Row"] == column_name, "Description"].values[0]
            return {"columns_description": description}
        else:
            return {"error": f"Column '{column_name}' does not exist in the DataFrame."}
    except Exception as e:
        return {"error": str(e)}

# Renvois la liste des colonne du df 
@app.get("/columns_names")
def columns_names(only_numeric: bool = False, exclude_id: bool = True):
    try:
        cols = df.columns.tolist()

        if exclude_id and "SK_ID_CURR" in cols:
            cols.remove("SK_ID_CURR")

        if only_numeric:
            cols = [c for c in cols if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]

        return {"columns_names": cols}
    except Exception as e:
        return {"error": str(e)}


# Prend un id et le nom d'une colopnne et renvois la vlauer de la colonne pour ce client et les valeurs de cette colonne pour les autres clients 
@app.get("/customer_specific_info")
def customer_specific_info(sk_id_curr: str, column_name: str):
    try:
        if column_name in df.columns and is_numeric_dtype(df[column_name]):
            customer_data = df[df["SK_ID_CURR"] == int(sk_id_curr)]
            
            if not customer_data.empty:
                column_values_list  = customer_data[column_name]
                column_value = column_values_list.values[0]
                # Série population -> numérique, puis liste JSON-safe
                ser_all = pd.to_numeric(df[column_name], errors="coerce")
                all_vals = []
                for v in ser_all.tolist():
                    try:
                        x = float(v)
                        all_vals.append(None if (math.isnan(x) or math.isinf(x)) else x)
                    except Exception:
                        all_vals.append(None)

                # Valeur client -> float safe
                col_ser = pd.to_numeric(customer_data[column_name], errors="coerce")
                column_value = None
                if len(col_ser):
                    try:
                        x = float(col_ser.values[0])
                        column_value = None if (math.isnan(x) or math.isinf(x)) else x
                    except Exception:
                        column_value = None

                # Moyenne safe (ignore NaN)
                mean_val = float(np.nanmean(ser_all)) if len(ser_all) else None
                if mean_val is not None and (math.isnan(mean_val) or math.isinf(mean_val)):
                    mean_val = None

                payload = {
                    "customer_value": column_value,
                    "all_customers_values": all_vals,
                    "mean_value": mean_val,
                }
                return json_safe(payload)

                return {
                    "customer_value": column_value,
                    
                    "all_customers_values": df[column_name].tolist(),
                    "mean_value": df[column_name].mean(),
                }
            else:
                return {"error": "Client not found"}
        else:
            return {"error": f"Column '{column_name}' does not exist in the DataFrame."}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/global_importance")
def global_importance(k: int = 20, method: str = "model", grouped: bool = True, shap_sample: int = 1000):
    """
    Importance globale des variables.
    - method="model": utilise feature_importances_ du LGBM (rapide, sans transformer les données)
    - method="shap" : mean(|SHAP|) sur un échantillon (plus fidèle)
    - grouped=True : agrège les OneHot sous leur variable d'origine
    """
    try:
        if df is None or df.empty:
            raise HTTPException(status_code=500, detail="Données non chargées")
        if modele is None:
            raise HTTPException(status_code=500, detail="Modèle non initialisé")

        pipeline = modele.get("pipeline", None)
        if pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline non présent dans le pickle")

        pre, est = _split_pipeline(pipeline)

        # Noms "bruts" (avant prétraitement)
        raw_names = FEATURE_ORDER if FEATURE_ORDER is not None else [c for c in df.columns if c != ID_COLUMN]

        method = method.lower().strip()

        # ---------- Méthode rapide: importances du modèle ----------
        if method == "model":
            # 1) valeurs d’importance
            if hasattr(est, "feature_importances_"):
                imps = np.asarray(est.feature_importances_, dtype=float)
            elif hasattr(est, "booster_"):  # certains wrappers LightGBM
                imps = np.asarray(est.booster_.feature_importance(), dtype=float)
            else:
                raise HTTPException(status_code=501, detail="Le modèle ne fournit pas feature_importances_")

            # 2) noms après prétraitement SANS transformer X (pas besoin)
            feat_names = _get_feature_names_after_preproc(pre, raw_names)

            # 3) alignement tailles (fallback générique si mismatch)
            if len(feat_names) != len(imps):
                logger.warning(f"Mismatch names({len(feat_names)}) vs imps({len(imps)}); fallback sur noms génériques.")
                feat_names = [f"feature_{i}" for i in range(len(imps))]

            # 4) agrégation par variable brute si demandé
            base_names = _map_transformed_to_raw(feat_names, raw_names) if grouped else feat_names

            agg = {}
            for name, val in zip(base_names, imps):
                agg[name] = agg.get(name, 0.0) + float(val)

            items = [{"name": n, "importance": v} for n, v in agg.items()]
            items.sort(key=lambda x: x["importance"], reverse=True)
            return {"items": items[:k], "method": "model", "grouped": grouped}

       
        else:
            raise HTTPException(status_code=400, detail="Paramètre 'method' doit être 'model' ou 'shap'.")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur /global_importance")
        raise HTTPException(status_code=500, detail=str(e))