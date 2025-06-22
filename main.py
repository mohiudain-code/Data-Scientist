
from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import joblib
from sqlmodel import Session

# Define the database model
class Prediction(SQLModel, table=True):
    __tablename__ = "Heart_Disease" 
    id: Optional[int] = Field(default=None, primary_key=True)
    age: int
    sex: int
    dataset: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalch: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    target: str

# Define the Pydantic model for request body
class TextData(BaseModel):
    age: int
    sex: int
    dataset: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalch: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Database connection URL
DataBase_url = ('postgresql://postgres:1122@localhost/Heart_disease')

# Create engine
engine = create_engine(DataBase_url, echo=True)

# Create tables
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# Get session for database operations
def get_session():
    with Session(engine) as s:
        yield s

# Load the pre-trained model
model = joblib.load('random_forest_model.joblib')

app = FastAPI()

@app.on_event('startup')
def on_startup():
    create_db_and_tables()

@app.post('/predict/')
async def predict_sentiment(data: TextData, session: Session = Depends(get_session)):
    # Make prediction using the input data
    prediction = model.predict([[
        data.age, data.sex, data.dataset, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg, data.thalch, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    # Assign target label based on prediction
    if prediction[0] == 0:
        target = 'No heart disease'
    elif prediction[0] == 1:
        target = 'Mild disease'
    elif prediction[0] == 2:
        target = 'Moderate heart disease'
    elif prediction[0] == 3:
        target = 'Severe heart disease'
    else:
        target = 'Critical heart disease'

    # Create a new prediction entry
    prediction_entry = Prediction(
        age=data.age,
        sex=data.sex,
        dataset=data.dataset,
        cp=data.cp,
        trestbps=data.trestbps,
        chol=data.chol,
        fbs=data.fbs,
        restecg=data.restecg,
        thalch=data.thalch,
        exang=data.exang,
        oldpeak=data.oldpeak,
        slope=data.slope,
        ca=data.ca,
        thal=data.thal,
        target=target
    )

    # Add the entry to the database
    session.add(prediction_entry)
    session.commit()
    session.refresh(prediction_entry)


    # Return the prediction result and saved ID
    return {'prediction': target, 'id': prediction_entry.id}
