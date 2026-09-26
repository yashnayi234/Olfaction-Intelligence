"""
FastAPI backend server for OlfacNet Intelligence.

Provides REST API endpoints for odor prediction from SMILES strings.

Usage:
    python scripts/api_server.py
    # or
    uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import json

from olfacnet import config
from olfacnet.inference.predictor import OlfacPredictor
from olfacnet.data.utils import load_odor_categories


# ─── API Models ──────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the molecule")
    top_k: Optional[int] = Field(10, description="Number of top predictions to return")
    threshold: Optional[float] = Field(0.1, description="Minimum confidence threshold")


class OdorPrediction(BaseModel):
    odor: str
    confidence: float


class PredictResponse(BaseModel):
    smiles: str
    predictions: list[OdorPrediction]
    full_profile: dict[str, float]


class StatsResponse(BaseModel):
    model_version: str
    num_classes: int
    num_parameters: int
    odor_categories: list[str]
    training_metrics: dict


# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="OlfacNet Intelligence API",
    description="Olfactory AI — Giving computers the sense of smell",
    version="2.0.0",
)

# CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor (lazy loaded)
_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = OlfacPredictor()
    return _predictor


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "OlfacNet Intelligence API",
        "version": "2.0.0",
        "description": "Olfactory AI — Predict odor profiles from molecular structures",
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict_odor(request: PredictRequest):
    """Predict odor profile for a molecule given its SMILES string."""
    try:
        predictor = get_predictor()
        
        # Get top-K predictions
        results = predictor.predict(
            request.smiles,
            top_k=request.top_k,
            threshold=request.threshold,
        )
        
        # Get full profile
        full_profile = predictor.predict_full(request.smiles)
        
        if not full_profile:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid SMILES string: '{request.smiles}'"
            )
        
        predictions = [
            OdorPrediction(odor=odor, confidence=conf)
            for odor, conf in results
        ]
        
        return PredictResponse(
            smiles=request.smiles,
            predictions=predictions,
            full_profile=full_profile,
        )
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/odors")
async def list_odors():
    """List all 105 odor categories."""
    categories = load_odor_categories()
    return {"categories": categories, "count": len(categories)}


@app.get("/api/stats", response_model=StatsResponse)
async def model_stats():
    """Get model statistics and training metrics."""
    predictor = get_predictor()
    categories = load_odor_categories()
    
    # Load training metrics if available
    metrics = {}
    metrics_path = os.path.join(config.RESULTS_DIR, "test_results.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    
    return StatsResponse(
        model_version="2.0.0",
        num_classes=config.NUM_CLASSES,
        num_parameters=predictor.model.count_parameters(),
        odor_categories=categories,
        training_metrics=metrics,
    )


@app.get("/api/examples")
async def example_molecules():
    """Get example molecules for testing."""
    return {
        "examples": [
            {"name": "Ethanol", "smiles": "CCO", "description": "Common alcohol"},
            {"name": "Vanillin", "smiles": "O=Cc1ccc(O)c(OC)c1", "description": "Vanilla fragrance"},
            {"name": "Limonene", "smiles": "CC(=C)C1CCC(=CC1)C", "description": "Citrus scent"},
            {"name": "Menthol", "smiles": "CC(C)C1CCC(C)CC1O", "description": "Minty coolness"},
            {"name": "Benzaldehyde", "smiles": "O=Cc1ccccc1", "description": "Almond/cherry"},
            {"name": "Eugenol", "smiles": "COc1cc(CC=C)ccc1O", "description": "Clove oil"},
            {"name": "Linalool", "smiles": "CC(=CCC/C(=C\\C)C)O", "description": "Floral lavender"},
            {"name": "Methanol", "smiles": "CO", "description": "Wood alcohol"},
            {"name": "Acetic Acid", "smiles": "CC(=O)O", "description": "Vinegar"},
            {"name": "Camphor", "smiles": "CC1(C)C2CCC1(C)C(=O)C2", "description": "Medicinal/woody"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
