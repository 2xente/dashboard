from fastapi import FastAPI
import pandas as pd
from backend import methode 

## Charger les données de test 
#df=pd.read_csv("s3://bixentep7/data_test_with_index.csv")
df= methode.load_data()

## Charger les données de test 
df_description =pd.read_csv("s3://bixentep7/HomeCredit_columns_description.csv", encoding='latin1')


## Charger le model
path = "s3://bixentep7/LGBMClassifier.pkl"
modele = pd.read_pickle(path) 



app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon API FastAPI 🚀"}

@app.get("/load_data")
def load_data():
    df=pd.read_csv("s3://bixentep7/data_test_with_index.csv")
    return {"data": df.to_dict(orient="records")}

@app.get("/load_model")
def load_model():
    path = "s3://bixentep7/LGBMClassifier.pkl"
    modele = pd.read_pickle(path)
    print(modele)
    return {
        "model": {
            "selector": modele["selector"],
            "feature_order": modele.get("feature_order", None),
            "model": modele["model"]
        }
    }

# a test 
@app.get("/prediction")
def prediction(sk_id_curr = str):
    """
    Fonction pour faire des prédictions sur une liste d'identifiants clients 
    """
    try:
        data_test = df[df["SK_ID_CURR"] == int(sk_id_curr)].drop("SK_ID_CURR", axis=1)
        transformed_data = methode.preprocess_data(data_test)
        selector = modele["selector"]
        model = modele["model"]
        feature_order = modele.get("feature_order", None)
        if feature_order is not None:
            data_test = data_test[feature_order]
        result = model.predict(data_test)
        return {"predictions": result.tolist()}
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
def columns_names():
    """
    Fonction pour obtenir la liste des noms de colonnes du DataFrame
    """
    try:
        column_names = df.columns.tolist()
        return {"columns_names": column_names}
    except Exception as e:
        return {"error": str(e)}

# Prend un id et le nom d'une colopnne et renvois la vlauer de la colonne pour ce client et les valeurs de cette colonne pour les autres clients 
@app.get("/customer_specific_info")
def customer_specific_info(sk_id_curr: str, column_name: str):
    try:
        if column_name in df.columns:
            customer_data = df[df["SK_ID_CURR"] == int(sk_id_curr)]
            if not customer_data.empty:
                column_value = customer_data[column_name].values[0]
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
