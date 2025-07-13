from pydantic import BaseModel, Field

class ComputeRequest(BaseModel):
    simulation_id: int = Field(..., alias="simulationId")
    room_temp: float = Field(..., alias="roomTemp")
    outdoor_temp: float = Field(..., alias="outdoorTemp")
    setpoint_temp: float = Field(..., alias="setpointTemp")

class ComputeReply(BaseModel):
    heaterPower: float
