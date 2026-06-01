from pydantic import BaseModel


class ClaimInput(BaseModel):
    Claim_Amount: float
    Patient_Age: int
    Patient_Gender: str
    Patient_City: str = "CityA"
    Patient_State: str = "CA"
    Hospital_ID: int = 1001
    Provider_Type: str
    Provider_Specialty: str
    Provider_City: str = "CityB"
    Provider_State: str = "CA"
    Diagnosis_Code: str = "D100"
    Procedure_Code: int = 10001
    Number_of_Procedures: int = 2
    Admission_Type: str
    Discharge_Type: str = "Home"
    Length_of_Stay_Days: int = 60
    Service_Type: str
    Deductible_Amount: float = 1200.0
    CoPay_Amount: float = 200.0
    Number_of_Previous_Claims_Patient: int = 1
    Number_of_Previous_Claims_Provider: int = 4
    Provider_Patient_Distance_Miles: float
    Claim_Submitted_Late: int
    Claim_Service_Gap_Days: int = 12
    Days_To_Policy_Expiry: int = 50
    Claim_Month: int = 6
    Claim_DayOfWeek: int = 2


class PredictionOutput(BaseModel):
    label: str
    probability: float
    confidence: str
    threshold_used: float
