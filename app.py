import os
import time
import base64
import hashlib
import numpy as np
import cv2
import face_recognition
from fastapi import FastAPI, Request
from fastapi.responses import Response, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

MASTER_CODE = "MASTER-REGISTER"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

CAPTURE_GUIDE = [
    "Hadapkan wajah ke kamera (frontal).",
    "Putar kepala ke kiri.",
    "Putar kepala ke kanan.",
]

# Ambang orientasi & kualitas
YAW_FRONT_MAX = 0.18
YAW_SIDE_MIN  = 0.28
MIN_SHARP     = 60.0

# Urutan langkah: 0=front,1=left,2=right
LABELS = ["front_1.jpg", "left_1.jpg", "right_1.jpg"]

state = {
    "mode": "IDLE",
    "current_user": None,
    "capture_step": 0,        # 0..2
    "captured_jpegs": [],     # pemenang per langkah (3)
    "pending_preview": False,
    "preview_expires_at": 0.0,
    "session_id": 0,
    "extras": [],             # dua terbaik global selain pemenang langkah
}

# -------------------------- Utils --------------------------
def user_dir(uid: str):
    p = os.path.join(SAVE_DIR, uid)
    os.makedirs(p, exist_ok=True)
    return p

def has_saved(uid: str) -> bool:
    p = user_dir(uid)
    return all(os.path.exists(os.path.join(p, name)) for name in LABELS)

def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _clear_user_jpg(uid: str):
    p = user_dir(uid)
    for name in os.listdir(p):
        if name.lower().endswith(".jpg"):
            try: os.remove(os.path.join(p, name))
            except: pass

def _save_now(uid: str):
    """Simpan segera: 3 orientasi + 2 ekstra terbaik (tanpa subfolder sesi)."""
    if not uid or len(state["captured_jpegs"]) < 3:
        return
    _clear_user_jpg(uid)
    p = user_dir(uid)

    # 3 orientasi (urut: front,left,right)
    for i, name in enumerate(LABELS):
        with open(os.path.join(p, name), "wb") as f:
            f.write(state["captured_jpegs"][i])

    # 2 ekstra terbaik non-duplikat
    extras_sorted = sorted(state["extras"], key=lambda x: x["score"], reverse=True)
    extras_sorted = extras_sorted[:2]
    for idx, ex in enumerate(extras_sorted, start=1):
        with open(os.path.join(p, f"extra_{idx}.jpg"), "wb") as f:
            f.write(ex["bytes"])

def _new_session():
    state["capture_step"] = 0
    state["captured_jpegs"].clear()
    state["pending_preview"] = False
    state["preview_expires_at"] = 0.0
    state["extras"].clear()
    state["session_id"] = int(time.time() * 1000)

def _enter_register():
    state["mode"] = "REGISTER"
    state["current_user"] = None
    _new_session()

def _enter_idle():
    state["mode"] = "IDLE"
    state["current_user"] = None
    _new_session()

def _build_ui(message: str = ""):
    payload = {
        "mode": state["mode"],
        "current_user": state["current_user"],
        "capture_step": state["capture_step"],
        "guide": CAPTURE_GUIDE[state["capture_step"]] if (state["mode"] == "REGISTER" and state["current_user"] and state["capture_step"] < 3) else "",
        "message": message,
        "preview": ["", "", ""],
        "reask_text": "",
        "session_id": state["session_id"],
    }
    if state["pending_preview"] and state["captured_jpegs"]:
        payload["preview"] = [
            "/buffer/0.jpg" if len(state["captured_jpegs"]) > 0 else "",
            "/buffer/1.jpg" if len(state["captured_jpegs"]) > 1 else "",
            "/buffer/2.jpg" if len(state["captured_jpegs"]) > 2 else "",
        ]
        if time.time() < state["preview_expires_at"]:
            payload["preview_deadline_ms"] = int((state["preview_expires_at"] - time.time()) * 1000)
            payload["reask_text"] = "Selesai. Data tersimpan. Scan user sama (≤10s) untuk ulang atau scan user lain."
    return payload

# -------------------------- Scoring --------------------------
_face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

def _decode_b64_to_bgr(data_url: str):
    if isinstance(data_url, dict):
        data_url = data_url.get("image", "")
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        return None, None
    try:
        head, b64 = data_url.split(",", 1)
        jpg_bytes = base64.b64decode(b64)
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr, jpg_bytes
    except Exception:
        return None, None

def _score_frame(bgr):
    if bgr is None:
        return -1.0, 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    faces = _face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return sharp * 0.2, sharp
    h, w = gray.shape[:2]
    face_area = max((fw * fh) for (_, _, fw, fh) in faces)
    score = sharp + (face_area / max(1.0, w * h)) * 1200.0
    return score, sharp

def _orient_match(yaw: float, step: int) -> bool:
    if step == 0:  # front
        return abs(yaw) <= YAW_FRONT_MAX
    if step == 1:  # left
        return yaw <= -YAW_SIDE_MIN
    if step == 2:  # right
        return yaw >= YAW_SIDE_MIN
    return False

def _push_extra(score: float, jpg: bytes, exclude_hashes: set):
    if jpg is None:
        return
    sha = _sha1(jpg)
    if sha in exclude_hashes:
        return
    for e in state["extras"]:
        if e["sha"] == sha:
            return
    state["extras"].append({"score": score, "sha": sha, "bytes": jpg})
    state["extras"].sort(key=lambda x: x["score"], reverse=True)
    if len(state["extras"]) > 2:
        state["extras"] = state["extras"][:2]

# -------------------------- Routes --------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "master": MASTER_CODE})

@app.get("/ui_state")
async def ui_state():
    return _build_ui("")

@app.post("/scan")
async def scan_route(data: dict):
    code = (data.get("code") or "").strip()
    if not code:
        return _build_ui("Kode kosong.")

    now = time.time()

    # MASTER: toggling mode (tidak diperlukan untuk simpan karena simpan otomatis)
    if code == MASTER_CODE:
        if state["mode"] == "IDLE":
            _enter_register()
            return _build_ui("REGISTER aktif. Scan user.")
        else:
            _enter_idle()
            return _build_ui("Kembali ke IDLE.")
    
    # Harus di REGISTER untuk menerima user
    if state["mode"] != "REGISTER":
        return _build_ui("Bukan REGISTER. Scan MASTER untuk masuk.")

    scanned_user = code

    # Jika user yang sama discan lagi saat preview aktif (≤10s) → ulang sesi bersih
    if state["pending_preview"] and state["current_user"] == scanned_user and now < state["preview_expires_at"]:
        _new_session()
        state["current_user"] = scanned_user
        return _build_ui(f"Ulang capture untuk {scanned_user}. Ikuti panduan.")

    # Beralih ke user baru: mulai sesi baru (data user sebelumnya sudah otomatis tersimpan saat step=3)
    state["current_user"] = scanned_user
    _new_session()

    # Tampilkan pratinjau set tersimpan bila sudah ada
    if has_saved(scanned_user):
        ui = _build_ui(f"Ditemukan data tersimpan untuk {scanned_user}. Tampilkan 10 detik.")
        ui["preview"] = [f"/saved/{scanned_user}/{i}.jpg" for i in range(3)]
        ui["reask_text"] = "Scan lagi (≤10s) untuk mulai ulang."
        state["pending_preview"] = True
        state["preview_expires_at"] = now + 10
        return ui

    return _build_ui(f"Mulai capture untuk {scanned_user}. Ikuti panduan tiap langkah.")

@app.post("/capture_auto")
async def capture_auto(data: dict):
    """
    Body: { session_id: <int>, images: [ {image: <dataURL>, yaw: <float>} , ... ] }
    Memilih satu terbaik per langkah; menyimpan dua ekstra terbaik global (non-duplicated).
    Setelah langkah ke-3 selesai → SIMPAN OTOMATIS ke /saved/<uid>/.
    """
    if state["mode"] != "REGISTER" or not state["current_user"]:
        return _build_ui("Tidak dapat capture: bukan REGISTER atau belum ada user.")
    if state["capture_step"] >= 3:
        return _build_ui("Capture lengkap.")

    client_sid = data.get("session_id", None)
    if client_sid is None or int(client_sid) != int(state["session_id"]):
        return _build_ui("Batch diabaikan (sesi berubah).")

    items = data.get("images", [])
    if not isinstance(items, list) or len(items) == 0:
        return _build_ui("Batch kosong.")

    step = state["capture_step"]

    candidates = []
    for it in items:
        if not isinstance(it, dict):
            continue
        yaw = it.get("yaw", None)
        if yaw is None or not _orient_match(float(yaw), step):
            continue
        bgr, raw = _decode_b64_to_bgr(it.get("image", ""))
        if bgr is None:
            continue
        score, sharp = _score_frame(bgr)
        if sharp < MIN_SHARP:
            continue
        candidates.append((score, sharp, raw))

    if not candidates:
        return _build_ui("Orientasi/ketajaman belum sesuai. Ulangi posisi sesuai panduan.")

    # pemenang langkah
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_bytes = candidates[0][2]
    state["captured_jpegs"].append(best_bytes)
    state["capture_step"] += 1

    # dua ekstra terbaik (global) kecuali pemenang
    exclude = {_sha1(best_bytes)}
    for score, _, raw in candidates[1:]:
        _push_extra(score, raw, exclude)
        if len(state["extras"]) >= 2:
            break

    # Jika sudah lengkap 3 langkah → SIMPAN OTOMATIS
    if state["capture_step"] == 3:
        uid = state["current_user"]
        _save_now(uid)
        state["pending_preview"] = True
        state["preview_expires_at"] = time.time() + 10
        return _build_ui("Tiga sudut lengkap. Data tersimpan. Pratinjau tampil 10 detik.")

    return _build_ui("Langkah tersimpan. Lanjut ke panduan berikutnya.")

@app.get("/buffer/{index}.jpg")
async def buffer_image(index: int):
    if 0 <= index < len(state["captured_jpegs"]):
        return Response(content=state["captured_jpegs"][index], media_type="image/jpeg")
    return Response(status_code=404)

@app.get("/saved/{uid}/{index}.jpg")
async def get_saved(uid: str, index: int):
    mapping = LABELS  # 0→front_1.jpg, 1→left_1.jpg, 2→right_1.jpg
    if not (0 <= index < 3):
        return Response(status_code=404)
    path = os.path.join(SAVE_DIR, uid, mapping[index])
    if os.path.exists(path):
        return FileResponse(path)
    return Response(status_code=404)

@app.post("/recognize")
async def recognize(data: dict):
    image_b64 = data.get("image", "")
    bgr, _ = _decode_b64_to_bgr(image_b64)
    if bgr is None:
        return {"result": "invalid"}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)
    if not encs:
        return {"result": "no_face"}
    query = encs[0]
    best_user = None
    best_dist = 0.6
    for uid in os.listdir(SAVE_DIR):
        ref_path = os.path.join(SAVE_DIR, uid, LABELS[0])
        if not os.path.exists(ref_path):
            continue
        img = face_recognition.load_image_file(ref_path)
        ref_enc = face_recognition.face_encodings(img)
        if not ref_enc:
            continue
        dist = face_recognition.face_distance(ref_enc, query)[0]
        if dist < best_dist:
            best_dist = dist
            best_user = uid
    if best_user:
        return {"result": "match", "user": best_user, "distance": float(best_dist)}
    return {"result": "unknown"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
