import numpy as np
from fastapi import HTTPException, status
from insightface.app import FaceAnalysis

from app.core.config import settings
from app.model.face_store import FaceStore
from app.model.schemas import FacePublic, FaceRecord, PredictResponse


class FaceController:
    def __init__(self) -> None:
        self.store = FaceStore(settings.json_db_path)
        self.face_app = FaceAnalysis(
            name=settings.insightface_model_name,
            providers=settings.insightface_providers,
        )
        self.face_app.prepare(
            ctx_id=settings.insightface_ctx_id,
            det_size=(settings.det_size, settings.det_size),
        )

    @staticmethod
    def _to_public(record: FaceRecord) -> FacePublic:
        return FacePublic(
            id=record.id,
            name=record.name,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32)
        denom = float(np.linalg.norm(vector))
        if denom <= 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Invalid face embedding",
            )
        return vector / denom

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray:
        faces = self.face_app.get(image)
        if not faces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in image",
            )

        best_face = max(
            faces,
            key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])),
        )
        embedding = getattr(best_face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(best_face, "embedding", None)
        if embedding is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unable to extract face embedding",
            )
        return self._normalize_embedding(embedding)

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(left, right) / ((np.linalg.norm(left) * np.linalg.norm(right)) + 1e-8))

    def register_face(self, name: str, image: np.ndarray) -> FacePublic:
        embedding = self._extract_embedding(image)
        record = self.store.create_face(name=name, embedding=embedding.tolist())
        return self._to_public(record)

    def list_faces(self) -> list[FacePublic]:
        records = self.store.list_faces()
        return [self._to_public(record) for record in records]

    def get_face(self, face_id: str) -> FacePublic:
        record = self.store.get_face(face_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face not found",
            )
        return self._to_public(record)

    def predict_face(self, image: np.ndarray, threshold: float | None = None) -> PredictResponse:
        limit = threshold if threshold is not None else settings.match_threshold
        query = self._extract_embedding(image)

        records = self.store.list_faces()
        if not records:
            return PredictResponse(matched=False, threshold=limit)

        best_score = -1.0
        best_record: FaceRecord | None = None

        for record in records:
            candidate = self._normalize_embedding(np.asarray(record.embedding, dtype=np.float32))
            score = self._cosine_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None or best_score < limit:
            return PredictResponse(matched=False, threshold=limit, score=best_score)

        return PredictResponse(
            matched=True,
            threshold=limit,
            score=best_score,
            face=self._to_public(best_record),
        )

    def modify_face(self, face_id: str, name: str | None, image: np.ndarray | None) -> FacePublic:
        if name is None and image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide at least one field to update (name or image)",
            )

        embedding: list[float] | None = None
        if image is not None:
            embedding = self._extract_embedding(image).tolist()

        updated = self.store.update_face(face_id=face_id, name=name, embedding=embedding)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face not found",
            )
        return self._to_public(updated)

    def delete_face(self, face_id: str) -> bool:
        return self.store.delete_face(face_id)
