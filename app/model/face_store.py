import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from app.model.schemas import FaceRecord


class FaceStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._lock = Lock()
        self._ensure_db_file()

    def _ensure_db_file(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self.db_path.write_text('{"faces": []}', encoding="utf-8")

    def _load_data(self) -> dict:
        raw = self.db_path.read_text(encoding="utf-8").strip()
        if not raw:
            return {"faces": []}
        data = json.loads(raw)
        if "faces" not in data:
            data["faces"] = []
        return data

    def _save_data(self, data: dict) -> None:
        self.db_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _to_record(item: dict) -> FaceRecord:
        return FaceRecord.model_validate(item)

    def list_faces(self) -> list[FaceRecord]:
        with self._lock:
            data = self._load_data()
            return [self._to_record(item) for item in data["faces"]]

    def get_face(self, face_id: str) -> FaceRecord | None:
        with self._lock:
            data = self._load_data()
            for item in data["faces"]:
                if item["id"] == face_id:
                    return self._to_record(item)
        return None

    def create_face(self, name: str, embedding: list[float]) -> FaceRecord:
        now = datetime.now(timezone.utc)
        record = {
            "id": str(uuid4()),
            "name": name,
            "embedding": embedding,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        with self._lock:
            data = self._load_data()
            data["faces"].append(record)
            self._save_data(data)
        return self._to_record(record)

    def update_face(
        self,
        face_id: str,
        name: str | None = None,
        embedding: list[float] | None = None,
    ) -> FaceRecord | None:
        with self._lock:
            data = self._load_data()
            for item in data["faces"]:
                if item["id"] == face_id:
                    if name is not None:
                        item["name"] = name
                    if embedding is not None:
                        item["embedding"] = embedding
                    item["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._save_data(data)
                    return self._to_record(item)
        return None

    def delete_face(self, face_id: str) -> bool:
        with self._lock:
            data = self._load_data()
            original_count = len(data["faces"])
            data["faces"] = [item for item in data["faces"] if item["id"] != face_id]
            if len(data["faces"]) == original_count:
                return False
            self._save_data(data)
            return True
