import asyncio
import logging
from pathlib import Path
from typing import Dict

import httpx
import numpy as np

from app.config import SIM_API, STORE_API, BASE_DIR, TRAINER_TTL_SEC
from app.models.q_learning_agent import QLearningAgent

logger = logging.getLogger("rl_trainer")

async def create_simulation(room_id, controller_type, iterations, timestep_seconds):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{SIM_API}/simulations/train-rl",
            json={
                "roomId": room_id,
                "controllerType": controller_type,
                "iterations": iterations,
                "timestepSeconds": timestep_seconds,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        sim_id = (
            data.get("simulationId")
            or data.get("simulation_id")
            or data.get("id")
            or (data.get("data") or {}).get("id")
        )
        return sim_id

async def persist_model(
    sim_id: int,
    room_id: int,
    q_table: np.ndarray,
    total_return: float,
    avg_return: float,
    episode_log: list[dict],
    lr: float,
    gamma: float,
    eps: float,
) -> str:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{STORE_API}/models/room-models/{room_id}",
                json={
                    "type": "RL",
                    "description": (
                        f"| 2-D Q-table (sim {sim_id}) | "
                        f"lr={lr:.3f}, gamma={gamma:.3f}, eps={eps:.3f} |"
                        f"total={total_return:.1f}, avg={avg_return:.3f} |"
                    ),
                },
                timeout=10,
            )
            resp.raise_for_status()
            model_id = resp.json().get("id") or resp.json().get("modelId")
    except Exception as exc:
        logger.info("STORE_API error: %s", exc)
        model_id = "error"

    out_dir = BASE_DIR / f"room-{room_id}" / f"model-{model_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        np.save(out_dir / "q_table.npy", q_table)
        with open(out_dir / "reward.txt", "w", encoding="utf-8") as f:
            f.write(f"total_return={total_return:.4f}\n")
            f.write(f"avg_return={avg_return:.4f}\n")
            f.write(f"episodes={len(episode_log)}\n")
            f.write("log=" + repr(episode_log) + "\n")
    except Exception as exc:
        logger.info("Error saving model to disk: %s", exc)
    return str(model_id)
