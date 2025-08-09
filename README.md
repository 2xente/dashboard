# dashboard

Activer l'environnement virtuel:
source venv/bin/activate

Lancement backend:
uvicorn backend.main:app --reload --port 8000
http://127.0.0.1:8000

lancement front:
streamlit run frontend/app.py
http://localhost:8501

