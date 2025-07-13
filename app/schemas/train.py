from pydantic import BaseModel, Field

class TrainRequest(BaseModel):
    room_id: int = Field(..., alias="roomId")
    iterations: int
    timestep_seconds: int = Field(..., alias="timestepSeconds")
    lr: float
    gamma: float
    eps: float

class TrainReply(BaseModel):
    message: str
    simulationId: int
    job_id: str
