// static/app.js
const mode = document.getElementById('mode');
const user = document.getElementById('user');
const step = document.getElementById('step');
const msg = document.getElementById('msg');
const p0 = document.getElementById('p0');
const p1 = document.getElementById('p1');
const p2 = document.getElementById('p2');
const reask = document.getElementById('reask');
const countdown = document.getElementById('countdown');
const faceBadge = document.getElementById('face');
const lastScan = document.getElementById('lastScan');
const video = document.getElementById('cam');
const overlay = document.getElementById('overlay');
const guideOverlay = document.getElementById('guideOverlay');
const guideTextEl = document.getElementById('guideText');
const ctx = overlay.getContext('2d');

let previewDeadline = 0;
let countdownTimer = null;
let currentUI = null;
let autoFlowRunning = false;

let hasFace = false;
let modelReady = false;

// tahan panduan ≥10s
let guideHoldUntil = 0;
let lastGuideText = "";

// session guard di sisi klien
let currentSessionId = 0;

function setMsg(t) { msg.textContent = t || ""; }
function updateFaceBadge() {
  faceBadge.innerHTML = hasFace
    ? '<span class="dot" style="background:#0a0;"></span>Face: Terdeteksi'
    : '<span class="dot" style="background:#a00;"></span>Face: Tidak Ada';
}
function syncCanvasSize() {
  const rect = video.getBoundingClientRect();
  overlay.width = Math.max(1, Math.round(rect.width));
  overlay.height = Math.max(1, Math.round(rect.height));
}
function toDataURLFromVideo(quality = 0.9) {
  const w = video.videoWidth;
  const h = video.videoHeight;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const cx = c.getContext('2d');
  cx.drawImage(video, 0, 0, w, h);
  return c.toDataURL('image/jpeg', quality);
}
function startCountdown() {
  if (countdownTimer) clearInterval(countdownTimer);
  countdownTimer = setInterval(() => {
    const remain = Math.max(0, previewDeadline - Date.now());
    const sec = Math.ceil(remain/1000);
    countdown.textContent = sec > 0 ? `Auto-hide: ${sec}s` : "";
    if (remain <= 0) {
      clearInterval(countdownTimer);
      fetch("/ui_state").then(r=>r.json()).then(setUI).catch(()=>{});
    }
  }, 200);
}

function setUI(s) {
  currentUI = s;
  currentSessionId = s.session_id || 0;

  mode.textContent = "Mode: " + s.mode;
  user.textContent = "User: " + (s.current_user || "-");
  step.textContent = "Step: " + s.capture_step + "/3";
  setMsg(s.message || "");

  const incomingGuide = (s.guide || "").trim();
  const now = Date.now();
  if (incomingGuide) {
    lastGuideText = incomingGuide;
    guideHoldUntil = now + 10_000;
  }
  const showGuide = (now < guideHoldUntil) && lastGuideText;
  guideTextEl.textContent = "Panduan: " + (showGuide ? lastGuideText : "-");
  guideOverlay.style.opacity = showGuide ? 1 : 0;

  if (s.preview) {
    p0.src = s.preview[0] || "";
    p1.src = s.preview[1] || "";
    p2.src = s.preview[2] || "";
  }
  reask.textContent = s.reask_text || "-";

  if (s.preview_deadline_ms) {
    previewDeadline = Date.now() + s.preview_deadline_ms;
    startCountdown();
  }

  maybeStartAutoFlow();
}

function refreshUI() {
  fetch("/ui_state").then(r=>r.json()).then(setUI).catch(()=>{});
}

function doScan(code) {
  lastScan.textContent = "Terakhir: " + code;
  fetch("/scan", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ code })
  }).then(r=>r.json()).then(setUI).catch(()=>{});
}

// ---------- Listener barcode (tanpa input) ----------
let scanBuffer = "";
let scanTimer = null;
const SCAN_TIMEOUT = 120;
function resetScanBuffer(){ scanBuffer=""; if (scanTimer){clearTimeout(scanTimer); scanTimer=null;} }
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.key === "Enter") {
    const code = scanBuffer.trim().toUpperCase();
    resetScanBuffer();
    if (code) doScan(code);
    e.preventDefault();
    return;
  }
  if (e.key && e.key.length === 1) {
    scanBuffer += e.key;
    if (scanTimer) clearTimeout(scanTimer);
    scanTimer = setTimeout(resetScanBuffer, SCAN_TIMEOUT);
  }
});

// ---------- Camera ----------
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(res => {
      const onLoaded = () => { video.removeEventListener('loadedmetadata', onLoaded); res(); };
      video.addEventListener('loadedmetadata', onLoaded);
    });
    await video.play();
    syncCanvasSize();
    new ResizeObserver(syncCanvasSize).observe(video.parentElement);
  } catch (e) {
    setMsg("Gagal membuka kamera: " + (e && e.message ? e.message : e));
  }
}

// ---------- face-api.js ----------
const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);
  modelReady = true;
}
function avgPoint(arr){ let x=0,y=0; for(const p of arr){x+=p.x;y+=p.y;} const n=arr.length||1; return {x:x/n,y:y/n}; }

async function detectOnce() {
  if (!modelReady || !video.srcObject) return [];
  const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.6 });
  const dets = await faceapi.detectAllFaces(video, opts).withFaceLandmarks(true);
  return (dets || []).map(d => {
    const lm = d.landmarks;
    const le = avgPoint(lm.getLeftEye());
    const re = avgPoint(lm.getRightEye());
    const mid = { x: (le.x + re.x)/2, y: (le.y + re.y)/2 };
    const nosePts = lm.getNose();
    const nose = nosePts[Math.min(3, nosePts.length-1)];
    const eyeDist = Math.hypot(le.x - re.x, le.y - re.y) || 1;
    const yawRaw = (nose.x - mid.x) / eyeDist;
    const yawView = -yawRaw; // mirror
    return { box: d.detection.box, score: d.detection.score, yaw: yawView };
  });
}

function drawOverlay(dets) {
  const vw = video.videoWidth || 1280;
  const vh = video.videoHeight || 720;
  const cw = overlay.width;
  const ch = overlay.height;
  const sx = cw / vw;
  const sy = ch / vh;

  ctx.clearRect(0, 0, cw, ch);
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0,255,0,0.9)";
  for (const d of dets) {
    const { x, y, width, height } = d.box;
    ctx.strokeRect(x * sx, y * sy, width * sx, height * sy);
  }
}

// ---------- Auto flow ----------
function canAutoFlow() {
  const inRegister = currentUI && currentUI.mode === "REGISTER";
  const hasUser = currentUI && !!currentUI.current_user;
  const notDone = currentUI && currentUI.capture_step < 3;
  const notAwaitSave = !(currentUI && currentUI.awaiting_save);
  return inRegister && hasUser && notDone && notAwaitSave;
}

async function captureBatch5s(sessionAtStart) {
  const durationMs = 5000;
  const intervalMs = 120;
  const maxFrames = 100;

  const frames = [];
  const t0 = performance.now();

  while (performance.now() - t0 < durationMs && frames.length < maxFrames) {
    if (currentSessionId !== sessionAtStart) break; // sesi berubah → hentikan batch
    const dets = await detectOnce();
    hasFace = dets.length > 0;
    updateFaceBadge();
    drawOverlay(dets);

    if (hasFace) {
      const best = dets.sort((a,b) => (b.score||0) - (a.score||0))[0];
      frames.push({
        image: toDataURLFromVideo(0.9),
        yaw: Number(best.yaw || 0)
      });
    }
    await new Promise(res => setTimeout(res, intervalMs));
  }
  return frames;
}

async function runAutoFlow() {
  autoFlowRunning = true;
  try {
    while (canAutoFlow()) {
      const sid = currentSessionId;
      const frames = await captureBatch5s(sid);
      if (!frames.length || sid !== currentSessionId) continue;

      const s = await fetch("/capture_auto", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ session_id: sid, images: frames })
      }).then(r=>r.json());

      setUI(s);
      if (!(s.mode === "REGISTER" && s.current_user)) break;
      if (s.capture_step >= 3) break;
    }
  } finally {
    autoFlowRunning = false;
  }
}

function maybeStartAutoFlow() {
  if (!autoFlowRunning && canAutoFlow()) runAutoFlow();
}

// ---------- Overlay loop ----------
function startOverlayLoop() {
  const loop = async () => {
    const dets = await detectOnce();
    hasFace = dets.length > 0;
    updateFaceBadge();
    drawOverlay(dets);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

// ---------- Boot ----------
(async function main() {
  await startCamera();
  await loadModels();
  startOverlayLoop();
  refreshUI();
  setInterval(refreshUI, 1200);
})();
