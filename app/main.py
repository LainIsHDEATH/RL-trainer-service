from fastapi import FastAPI
from app.routers.trainer_router import router as train_router
from app.logging_config import setup_logging

setup_logging()

app = FastAPI(title="RL Trainer")
app.include_router(train_router)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7002, reload=True)
