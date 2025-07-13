import os
from pathlib import Path

SIM_API = os.getenv("SIM_API", "http://localhost:8080/api")
STORE_API = os.getenv("STORE_API", "http://localhost:8082/api")
BASE_DIR = Path(os.getenv("MODELS_DIR", "../models"))
BASE_DIR.mkdir(parents=True, exist_ok=True)
TRAINER_TTL_SEC = 30 * 60
