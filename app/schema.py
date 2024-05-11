# === import pydantic 
from pydantic import BaseModel 




# Model Input Schema
class ModelInput(BaseModel) : 
    num_rooms : int 
    home_size : int 
    distance_from_sea : int 
# 