# === Fastapi Libs
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
 # === import pydantic 
from pydantic import BaseModel 




# Model Input Schema
class ModelInput(BaseModel) : 
    num_rooms : int 
    home_size : int 
    distance_from_sea : int 
# 


# === Model  
from .ML_model import predict_home_price

 
app = FastAPI() # endPoint

# Define allowed origins
origins = [
    "*"
]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Routes === # 
@app.get("/")
async def root(): 
    return {"Hello" : "ML Model V0.0"}

# model routes 
@app.post("/ml/")
async def ml_model(options : ModelInput ): 

    # Validate Options 
    if options.num_rooms <= 0 or options.home_size <=0 or options.distance_from_sea <=0 : 
        raise HTTPException(status_code=400 , detail="Invalid Input")
    
    # Predict home price 
    value = predict_home_price(options.num_rooms, options.home_size, options.distance_from_sea)
        
    return {"value" : value}

