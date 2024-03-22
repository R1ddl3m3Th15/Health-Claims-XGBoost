from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the trained XGBoost model from the pkl file
with open('xgboost_claim_prediction_model.pkl', 'rb') as file:
    xg_model = pickle.load(file)

# Initialize the FastAPI app
app = FastAPI()

# Define input data schema using Pydantic BaseModel


class PredictionInput(BaseModel):
    DeductibleAmtPaid: float
    RenalDiseaseIndicator: int
    NoOfMonths_PartACov: int
    NoOfMonths_PartBCov: int
    ChronicCond_Alzheimer: int
    ChronicCond_Heartfailure: int
    ChronicCond_KidneyDisease: int
    ChronicCond_Cancer: int
    ChronicCond_ObstrPulmonary: int
    ChronicCond_Depression: int
    ChronicCond_Diabetes: int
    ChronicCond_IschemicHeart: int
    ChronicCond_Osteoporasis: int
    ChronicCond_rheumatoidarthritis: int
    ChronicCond_stroke: int
    IPAnnualReimbursementAmt: int
    IPAnnualDeductibleAmt: int
    OPAnnualReimbursementAmt: int
    OPAnnualDeductibleAmt: int
    LengthOfStay: float
    ClaimDuration: int
    Age: int
    NumDiagnosisCodes: int
    NumProcedureCodes: int

# Define prediction endpoint


@app.post("/predict")
async def predict(insurance_data: PredictionInput):
    # Convert input data to DataFrame
    input_data = pd.DataFrame(insurance_data.dict(), index=[0])
    # Make prediction using the loaded model
    prediction = xg_model.predict(input_data)
    return {"prediction": prediction.tolist()}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# for testing, uvicorn main:app --host 0.0.0.0 --port 8000
