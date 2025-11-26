import numpy as np
from src.model import load_trained_model
from src.preprocessing import preprocess_image

class Predictor:
    """
    Wrapper class for model inference with Test Time Augmentation (TTA).
    """
    def __init__(self, model_path="./models/arcade_model.h5"):
        self.model = load_trained_model(model_path)

    def predict_image(self, image_bytes):
        if self.model is None:
            return {"error": "Model is not loaded. Check model path."}

        try:
            # 1. Preprocess
            input_tensor = preprocess_image(image_bytes)

            # SAFETY CHECK: Fix "Double Adapter" bug
            if input_tensor.shape[-1] == 3:
                input_tensor = input_tensor[..., :1]

            # 2. TTA (Test Time Augmentation) - The "Free Accuracy" Trick
            # Prediction A: Original Image
            pred_1 = self.model.predict(input_tensor, verbose=0)
            score_1 = float(pred_1[0][0])

            # Prediction B: Flipped Image
            # We flip the image horizontally. If the model is robust, 
            # the average of these two will be more accurate than either one alone.
            input_flipped = np.flip(input_tensor, axis=2)
            pred_2 = self.model.predict(input_flipped, verbose=0)
            score_2 = float(pred_2[0][0])

            # Average the scores
            final_score = (score_1 + score_2) / 2.0

            # 3. Interpretation
            is_stenosis = final_score > 0.5
            label = "Stenosis (Unhealthy)" if is_stenosis else "Structure (Healthy)"
            
            # Confidence calculation
            confidence = final_score if is_stenosis else (1 - final_score)

            return {
                "diagnosis": label,
                "confidence": confidence,
                "raw_score": final_score,
                "tta_details": {"original": score_1, "flipped": score_2}
            }
            
        except Exception as e:
            print(f"Prediction Error: {e}")
            return {"error": str(e)}