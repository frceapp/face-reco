import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    def __init__(self) -> None:
        self.api_token = os.getenv("API_TOKEN", "change-this-token")
        self.token_header_name = os.getenv("TOKEN_HEADER_NAME", "X-API-Token")

        db_path = os.getenv("JSON_DB_PATH", "data/faces.json")
        self.json_db_path = Path(db_path)
        if not self.json_db_path.is_absolute():
            self.json_db_path = BASE_DIR / self.json_db_path

        self.match_threshold = float(os.getenv("MATCH_THRESHOLD", "0.45"))
        self.insightface_model_name = os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l")
        providers = os.getenv("INSIGHTFACE_PROVIDERS", "CPUExecutionProvider")
        self.insightface_providers = [item.strip() for item in providers.split(",") if item.strip()]
        self.insightface_ctx_id = int(os.getenv("INSIGHTFACE_CTX_ID", "0"))
        self.det_size = int(os.getenv("DET_SIZE", "640"))


settings = Settings()
