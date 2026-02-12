from fastapi import FastAPI

from app.router.face_router import router as face_router

app = FastAPI(
    title="Face Recognition API",
    version="1.0.0",
    description="FastAPI + InsightFace",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(face_router, prefix="/api")
