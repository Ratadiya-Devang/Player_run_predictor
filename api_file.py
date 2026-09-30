from fastapi import FastAPI
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

model = joblib.load("player_score_prediction.pkl")


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Prediction API
@app.get("/prediction")
def prediction(
    m1: int,
    m2: int,
    m3: int,
    m4: int,
    m5: int
):

    new_data = pd.DataFrame({
        "match1": [m1],
        "match2": [m2],
        "match3": [m3],
        "match4": [m4],
        "match5": [m5],
    })

    runs = model.predict(new_data)

    r = int(round(runs[0]))

    return {"run": r}


# Test API
@app.get("/")
def testapi():
    return {
        "msg": "API Tested Successfully"
    }