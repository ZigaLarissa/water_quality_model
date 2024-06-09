from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import pickle as pk
import numpy as np
from sklearn.preprocessing import StandardScaler

# Create a FastAPI app
app = FastAPI()


# Define a model class for the request body
class WaterQualityRequest(BaseModel):
    ph: Union[float, None] = None
    hardness: Union[float, None] = None
    solids: Union[float, None] = None
    chloramines: Union[float, None] = None
    sulfate: Union[float, None] = None
    conductivity: Union[float, None] = None
    organic_carbon: Union[float, None] = None
    trihalomethanes: Union[float, None] = None
    turbidity: Union[float, None] = None


@app.get("/")
async def index():
    return {
        "message": "Yes, you are connected to the Internet and you are connected to api "
    }


@app.post("/water-quality")
async def water_quality(request: WaterQualityRequest):
    try:
        # Load the trained model
        with open("./water_potability_model.pkl", "rb") as file:
            model = pk.load(file)

        # Prepare the request data and turn it into
        data = [
            [
                request.ph,
                request.hardness,
                request.solids,
                request.chloramines,
                request.sulfate,
                request.conductivity,
                request.organic_carbon,
                request.trihalomethanes,
                request.turbidity,
            ]
        ]

        # Standardize the features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Make a prediction
        prediction = model.predict(data_scaled)
        prediction = np.round(prediction).astype(int)
        prediction = prediction[0][0]

        # Convert the prediction to a string
        # based on the prediction value
        # if the prediction is 1, the water is potable; otherwise, it is not potable
        if prediction == 1:
            prediction = "Potable"
        else:
            prediction = "Not Potable"

        return {"potability": prediction}
    except Exception as e:
        return {"error": str(e)}