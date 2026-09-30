from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
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


# Data received from React
class PredictionInput(BaseModel):
    m1: int
    m2: int
    m3: int
    m4: int
    m5: int


# POST prediction API
@app.post("/prediction")
def prediction(data: PredictionInput):

    new_data = pd.DataFrame({
        "match1": [data.m1],
        "match2": [data.m2],
        "match3": [data.m3],
        "match4": [data.m4],
        "match5": [data.m5],
    })

    runs = model.predict(new_data)

    r = int(round(runs[0]))

    return {"run": r}


# Test API
@app.get("/")
def testapi():
    return {"msg": "API Tested Successfully"}