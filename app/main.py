from fastapi import FastAPI
from app.routers.trainer_router import router as train_router
from app.logging_config import setup_logging
import asyncio
import logging

setup_logging()
logger = logging.getLogger("rl_trainer")

app = FastAPI(title="RL Trainer")
app.include_router(train_router)

app.state.trainers = {}  # именно здесь создаём state

from app.config import TRAINER_TTL_SEC

@app.on_event("startup")
async def _start_gc():
    asyncio.create_task(_trainer_gc(app))

async def _trainer_gc(app: FastAPI):
    while True:
        await asyncio.sleep(60)
        now = asyncio.get_event_loop().time()
        to_del = [sid for sid, e in app.state.trainers.items() if now - e["last_touch"] > TRAINER_TTL_SEC]
        for sid in to_del:
            logger.info("Trainer %s removed due to inactivity", sid)
            app.state.trainers.pop(sid, None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7002, reload=True)