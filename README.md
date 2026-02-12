# Face Recognition API

FastAPI application for face registration, prediction, modification, and deletion using InsightFace.
All face data is persisted in a JSON file.

## Project structure

```text
app/
  controller/
  core/
  deps/
  model/
  router/
  utils/
data/
main.py
.env
requirements.txt
```

## Setup

1. Create and activate virtual env.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Update `.env`:

- `API_TOKEN`: token required in request header
- `JSON_DB_PATH`: JSON storage file path
- `MATCH_THRESHOLD`: cosine similarity threshold

4. Run API:

```bash
uvicorn main:app --reload
```

## Auth header

Send token on every `/api/faces/*` request:

- Header name: `X-API-Token` (or value from `TOKEN_HEADER_NAME` in `.env`)
- Header value: value of `API_TOKEN`

## Endpoints

- `POST /api/faces/register` (multipart: `name`, `image`)
- `POST /api/faces/predict` (multipart: `image`, optional `threshold`)
- `PUT /api/faces/{face_id}` (multipart: optional `name`, optional `image`)
- `DELETE /api/faces/{face_id}`
- `GET /api/faces`
- `GET /api/faces/{face_id}`
- `GET /health`
