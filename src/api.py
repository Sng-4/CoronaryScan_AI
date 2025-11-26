from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from src.prediction import Predictor
import os
import time

# Initialize FastAPI app
app = FastAPI(
    title="ARCADE Stenosis Detector API",
    description="A medical AI API for detecting coronary artery stenosis from X-ray angiography.",
    version="1.0.0"
)

# Global Variables
# We load the predictor once at startup so we don't reload the heavy model for every request
MODEL_PATH = "models/arcade_model.h5"
predictor = None

@app.on_event("startup")
def load_model_on_startup():
    """
    Load the model when the API server starts.
    """
    global predictor
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Startup: Loading model from {MODEL_PATH}...")
        predictor = Predictor(MODEL_PATH)
        if predictor.model:
            print("✅ Startup: Model loaded successfully!")
        else:
            print("⚠️ Startup: Failed to load model architecture.")
    else:
        print(f"⚠️ Startup: Model file not found at {MODEL_PATH}. Predictions will fail until a model is trained.")

@app.get("/")
def health_check():
    """
    Simple health check to verify the API is running.
    """
    model_status = "active" if predictor and predictor.model else "inactive"
    return {
        "status": "online", 
        "model_status": model_status,
        "service": "ARCADE AI Diagnostics"
    }

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to predict stenosis from an uploaded image file.
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please contact admin or check server logs.")

    # Read file bytes
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # Run inference using our helper class
    result = predictor.predict_image(contents)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result

def simulated_retraining_task():
    """
    Simulates a long-running background retraining process.
    In a real system, this would trigger a training script or Kubeflow pipeline.
    """
    print("🔄 Background Task: Retraining pipeline triggered...")
    time.sleep(5) # Simulate delay
    print("✅ Background Task: Retraining simulation complete. New weights saved (simulated).")

@app.post("/retrain")
async def retrain_endpoint(background_tasks: BackgroundTasks):
    """
    Trigger a model retraining process in the background.
    """
    # We use BackgroundTasks so the API returns immediately while the heavy job runs
    background_tasks.add_task(simulated_retraining_task)
    return {"message": "Retraining pipeline triggered. The system will update shortly."}

if __name__ == "__main__":
    import uvicorn
    # This block allows you to run the file directly with 'python src/api.py'
    uvicorn.run(app, host="0.0.0.0", port=8000)