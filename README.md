# dashboard

Activer l'environnement virtuel:
source venv/bin/activate

Lancement backend sur http://127.0.0.1:8000
```
uvicorn backend.main:app --reload --port 8000
```

lancement front sur http://localhost:8501
```
streamlit run frontend/app.py
```

# Set up 

Configurer aws avec la commande suivante et les clés d'accées pour accéder au csv dans S3

```
aws configure
```


