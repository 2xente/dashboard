from fastapi import FastAPI
import streamlit as st


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon API FastAPI 🚀"}

@app.get("/hello/{name}")
def greet(name: str):
    return {"message": f"Hello {name} 👋"}

# Titre de l'application
st.title("Dashboard explication risque crédit")

# Zone de texte
SK_ID_CURR = st.text_input("Entrez l'identifiant client")

# Bouton
if st.button("Dire bonjour"):
    if SK_ID_CURR:
        st.success(f"Bonjour, {SK_ID_CURR} 👋")
    else:
        st.warning("Veuillez entrer un nom avant de cliquer.")
