from datetime import datetime

from pydantic import BaseModel, Field


class FaceRecord(BaseModel):
    id: str
    name: str
    embedding: list[float]
    created_at: datetime
    updated_at: datetime


class FacePublic(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime


class PredictResponse(BaseModel):
    matched: bool
    threshold: float
    score: float | None = None
    face: FacePublic | None = None


class MessageResponse(BaseModel):
    message: str


class FacesResponse(BaseModel):
    faces: list[FacePublic] = Field(default_factory=list)
