import sys
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal

from src.logger import logging
from src.exception import CustomException
from src.pipeline.predict_pipeline import PredictPipeline, DiamondData


# App setup
app = FastAPI(
    title="Diamond Price Predictor",
    description="Predicts diamond price based on physical properties",
    version="1.0.0"
)

# CORS — allows frontend/external requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema
class DiamondRequest(BaseModel):
    carat   : float = Field(..., gt=0, example=0.23)
    cut     : Literal['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'] = Field(..., example='Ideal')
    color   : Literal['J', 'I', 'H', 'G', 'F', 'E', 'D'] = Field(..., example='E')
    clarity : Literal['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'] = Field(..., example='SI2')
    depth   : float = Field(..., gt=0, example=61.5)
    table   : float = Field(..., gt=0, example=55.0)
    x       : float = Field(..., gt=0, example=3.95)
    y       : float = Field(..., gt=0, example=3.98)
    z       : float = Field(..., gt=0, example=2.43)


# Response schema
class DiamondResponse(BaseModel):
    predicted_price_usd : float
    model_version       : str = "1.0.0"
    status              : str = "success"


# Routes
@app.get("/")
def root():
    return {
        "message": "Diamond Price Predictor API",
        "docs"   : "/docs",
        "health" : "/health"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=DiamondResponse)
def predict(request: DiamondRequest):
    try:
        logging.info(f"Prediction request received: {request}")

        # Build input dataframe
        diamond = DiamondData(
            carat   = request.carat,
            cut     = request.cut,
            color   = request.color,
            clarity = request.clarity,
            depth   = request.depth,
            table   = request.table,
            x       = request.x,
            y       = request.y,
            z       = request.z
        )
        df = diamond.get_data_as_dataframe()

        # Run prediction
        pipeline = PredictPipeline()
        result   = pipeline.predict(df)

        predicted_price = round(float(result[0]), 2)
        logging.info(f"Predicted price: ${predicted_price:,.2f}")

        return DiamondResponse(
            predicted_price_usd=predicted_price
        )

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))