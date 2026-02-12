from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.controller.face_controller import FaceController
from app.deps.auth import require_api_token
from app.model.schemas import FacePublic, FacesResponse, MessageResponse, PredictResponse
from app.utils.image_utils import read_upload_image

router = APIRouter(prefix="/faces", tags=["faces"], dependencies=[Depends(require_api_token)])
face_controller = FaceController()


@router.post("/register", response_model=FacePublic)
async def register_face(
    name: str = Form(...),
    image: UploadFile = File(...),
) -> FacePublic:
    image_data = await read_upload_image(image)
    return face_controller.register_face(name=name, image=image_data)


@router.post("/predict", response_model=PredictResponse)
async def predict_face(
    image: UploadFile = File(...),
    threshold: float | None = Form(default=None),
) -> PredictResponse:
    image_data = await read_upload_image(image)
    return face_controller.predict_face(image=image_data, threshold=threshold)


@router.put("/{face_id}", response_model=FacePublic)
async def modify_face(
    face_id: str,
    name: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
) -> FacePublic:
    image_data = None
    if image is not None:
        image_data = await read_upload_image(image)
    return face_controller.modify_face(face_id=face_id, name=name, image=image_data)


@router.delete("/{face_id}", response_model=MessageResponse)
async def delete_face(face_id: str) -> MessageResponse:
    deleted = face_controller.delete_face(face_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face not found",
        )
    return MessageResponse(message="Face deleted")


@router.get("", response_model=FacesResponse)
async def list_faces() -> FacesResponse:
    return FacesResponse(faces=face_controller.list_faces())


@router.get("/{face_id}", response_model=FacePublic)
async def get_face(face_id: str) -> FacePublic:
    return face_controller.get_face(face_id)
