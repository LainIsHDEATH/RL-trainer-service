import uuid
import asyncio
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request
from app.schemas.train import TrainRequest, TrainReply
from app.schemas.compute import  ComputeRequest, ComputeReply

from app.models.q_learning_agent import QLearningAgent
from app.services.trainer_service import create_simulation, persist_model
from app.config import TRAINER_TTL_SEC
import logging

router = APIRouter()
logger = logging.getLogger("rl_trainer")

@router.post("/train", response_model=TrainReply)
async def train(req: TrainRequest, bg: BackgroundTasks, request: Request):
    app = request.app
    sim_id = await create_simulation(
        req.room_id, "TRAIN_RL", req.iterations, req.timestep_seconds
    )
    if sim_id in app.state.trainers:
        raise HTTPException(400, "Trainer already exists for this simulation")
    agent = QLearningAgent(
        n_bins=101,
        total_steps=req.iterations,
        lr=req.lr,
        gamma=req.gamma,
        eps=req.eps,
    )
    app.state.trainers[sim_id] = {
        "agent": agent,
        "room_id": req.room_id,
        "lr": req.lr,
        "gamma": req.gamma,
        "eps": req.eps,
        "prev_state": None,
        "prev_action": None,
        "prev_pct": None,
        "prev_setpoint": None,
        "last_touch": asyncio.get_event_loop().time(),
    }
    job_id = str(uuid.uuid4())
    logger.info("Trainer started for simulation %s (room %s)", sim_id, req.room_id)
    return {"message": "Training initialized", "simulationId": sim_id, "job_id": job_id}

@router.post("/compute", response_model=ComputeReply)
async def compute(step: ComputeRequest, request: Request):
    app = request.app
    entry = app.state.trainers.get(step.simulation_id)
    if not entry:
        raise HTTPException(404, "No active trainer for this simulation")
    now = asyncio.get_event_loop().time()
    entry["last_touch"] = now
    agent: QLearningAgent = entry["agent"]
    current_state = agent._state(step.room_temp, step.outdoor_temp)
    if entry["prev_state"] is not None:
        reward = -abs(entry["prev_setpoint"] - step.room_temp) - 0.04 * entry["prev_pct"]
        done = agent.learn(entry["prev_state"], entry["prev_action"], reward, current_state)
        if done:
            await _finish_training(step.simulation_id, entry)
            pct = 0.0
            return ComputeReply(heaterPower=pct * 100)
    pct, state_bins, action_bin = agent.act(step.room_temp, step.outdoor_temp)
    entry["prev_state"] = state_bins
    entry["prev_action"] = action_bin
    entry["prev_pct"] = pct
    entry["prev_setpoint"] = step.setpoint_temp
    return ComputeReply(heaterPower=pct * 100)

async def _finish_training(sim_id: int, entry: dict) -> None:
    agent: QLearningAgent = entry["agent"]
    room_id = entry["room_id"]
    model_id = await persist_model(
        sim_id,
        room_id,
        agent.q_table,
        agent.last_total or 0.0,
        agent.last_avg or 0.0,
        agent.returns,
        entry["lr"],
        entry["gamma"],
        entry["eps"],
    )
    logger.info("Model %s for room %s saved, trainer removed", model_id, room_id)
    entry.pop(sim_id, None)
