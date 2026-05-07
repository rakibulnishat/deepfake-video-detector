import os
import io
import base64
import subprocess
import tempfile
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

# ─── DEVICE ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FACE_SIZE  = 224
NUM_FRAMES = 8
STA_HIDDEN = 256
BANK_SIZE  = 512
DROPOUT    = 0.3

# ─── MODEL ────────────────────────────────────────────────────────────────────

class SpatiotemporalAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.spatial_proj  = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.temporal_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
    def forward(self, x):
        return self.spatial_proj(x), self.temporal_proj(x)


class ContrastiveFeatureBank(nn.Module):
    def __init__(self, feature_dim, bank_size=512):
        super().__init__()
        self.register_buffer("real_bank", torch.randn(bank_size, feature_dim))
        self.register_buffer("fake_bank", torch.randn(bank_size, feature_dim))
        self.register_buffer("real_ptr",  torch.zeros(1, dtype=torch.long))
        self.register_buffer("fake_ptr",  torch.zeros(1, dtype=torch.long))


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=False, num_classes=0, global_pool="avg")
        self.out_dim = self.backbone.num_features
    def forward(self, x):
        return self.backbone(x)


class TemporalEncoder(nn.Module):
    def __init__(self, spatial_dim, num_frames=8):
        super().__init__()
        swin_dim         = 768
        self.input_proj  = nn.Linear(spatial_dim, swin_dim)
        self.pos_embed   = nn.Parameter(torch.randn(1, num_frames, swin_dim) * 0.02)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=swin_dim, nhead=8, dim_feedforward=swin_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_dim     = swin_dim
    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1)]
        return self.transformer(x).mean(dim=1)


class DeepfakeDetector(nn.Module):
    def __init__(self, num_frames=8, sta_hidden=256, bank_size=512, dropout=0.3):
        super().__init__()
        self.num_frames   = num_frames
        self.spatial_enc  = SpatialEncoder()
        self.temporal_enc = TemporalEncoder(self.spatial_enc.out_dim, num_frames)
        fused_dim         = self.spatial_enc.out_dim + self.temporal_enc.out_dim
        self.sta          = SpatiotemporalAdapter(fused_dim, sta_hidden)
        self.bank         = ContrastiveFeatureBank(sta_hidden * 2, bank_size)
        self.classifier   = nn.Sequential(
            nn.Linear(sta_hidden * 2, 256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 1))
    def forward(self, frames):
        B, T, C, H, W = frames.shape
        sp_flat        = self.spatial_enc(frames.view(B * T, C, H, W))
        spatial        = sp_flat.view(B, T, -1)
        temporal       = self.temporal_enc(spatial)
        fused          = torch.cat([spatial.mean(1), temporal], dim=-1)
        s, t           = self.sta(fused)
        combined       = torch.cat([s, t], dim=-1)
        return torch.sigmoid(self.classifier(combined).squeeze(-1))


# ─── LOAD MODEL ───────────────────────────────────────────────────────────────

def load_model():
    model = DeepfakeDetector(NUM_FRAMES, STA_HIDDEN, BANK_SIZE, DROPOUT)
    ckpt  = torch.load("best_model.pt", map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)

MODEL = load_model()
print(f"Model loaded on {DEVICE}")

# ─── TRANSFORMS ───────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((FACE_SIZE, FACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── FRAME EXTRACTION (supports all mobile formats via ffmpeg) ────────────────

def extract_frames(video_path):
    """
    Extract frames from any video format.
    Uses ffmpeg to convert mobile formats (HEVC, MOV, HEIC) to H264 first.
    Falls back to direct OpenCV read if conversion fails.
    """
    converted_path = video_path + "_converted.mp4"
    read_path      = video_path

    try:
        result = subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-preset", "fast",
            "-y", converted_path
        ], capture_output=True, timeout=120)
        if os.path.exists(converted_path) and os.path.getsize(converted_path) > 0:
            read_path = converted_path
    except Exception:
        read_path = video_path

    cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        # Last resort: try original path
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if os.path.exists(converted_path):
            os.remove(converted_path)
        return None

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = NUM_FRAMES
    indices = np.linspace(0, max(total - 1, 0), NUM_FRAMES, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        m    = min(h, w)
        crop = rgb[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
        frames.append(Image.fromarray(crop))

    cap.release()
    if os.path.exists(converted_path):
        os.remove(converted_path)

    # Pad if not enough frames
    while len(frames) < NUM_FRAMES and len(frames) > 0:
        frames.append(frames[-1])

    return frames if len(frames) > 0 else None

# ─── INFERENCE ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(video_path):
    frames = extract_frames(video_path)
    if not frames:
        return None

    tensors = torch.stack([TRANSFORM(f) for f in frames]).unsqueeze(0).to(DEVICE)
    prob    = MODEL(tensors).item()
    conf    = max(prob, 1 - prob) * 100

    if prob >= 0.7:
        verdict, risk, color = "FAKE",        "HIGH RISK",   "#e63946"
    elif prob >= 0.5:
        verdict, risk, color = "LIKELY FAKE", "MEDIUM RISK", "#f4a261"
    elif prob >= 0.3:
        verdict, risk, color = "LIKELY REAL", "LOW RISK",    "#e9c46a"
    else:
        verdict, risk, color = "REAL",        "AUTHENTIC",   "#2dc653"

    # Build frame preview grid
    grid = Image.new("RGB", (4 * 160, 2 * 160), (20, 20, 30))
    for i, f in enumerate(frames[:8]):
        r, c = divmod(i, 4)
        grid.paste(f.resize((160, 160)), (c * 160, r * 160))
    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    grid_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "verdict":    verdict,
        "risk":       risk,
        "color":      color,
        "fake_pct":   round(prob * 100, 1),
        "real_pct":   round((1 - prob) * 100, 1),
        "confidence": round(conf, 1),
        "frames":     len(frames),
        "device":     DEVICE.upper(),
        "grid_b64":   grid_b64,
    }

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────

app = FastAPI()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepfake Video Detector</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0a0f;color:#e8e8f0;font-family:'Segoe UI',sans-serif;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:40px 20px}
h1{font-size:2.2rem;font-weight:800;margin-bottom:6px;letter-spacing:-0.02em}
h1 span{color:#e63946}
.sub{color:#6b6b8a;font-size:0.82rem;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:40px}
.card{background:#12121a;border:1px solid #1e1e2e;border-radius:14px;padding:32px;width:100%;max-width:680px;margin-bottom:24px}
.upload-area{border:2px dashed #2e2e4e;border-radius:10px;padding:40px;text-align:center;cursor:pointer;transition:border-color 0.2s;margin-bottom:20px}
.upload-area:hover{border-color:#e63946}
#fileInput{display:none}
.upload-icon{font-size:3rem;margin-bottom:12px}
.upload-text{color:#6b6b8a;font-size:0.9rem}
.upload-text strong{color:#e8e8f0}
.file-name{color:#e63946;font-size:0.85rem;margin-top:8px;word-break:break-all}
.btn{background:#e63946;color:white;border:none;border-radius:8px;padding:14px 32px;font-size:1rem;font-weight:700;cursor:pointer;width:100%;transition:opacity 0.2s}
.btn:hover{opacity:0.85}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.result-card{background:#12121a;border:1px solid #1e1e2e;border-radius:14px;padding:32px;width:100%;max-width:680px;display:none}
.verdict{font-size:2rem;font-weight:800;text-align:center;padding:20px;border-radius:10px;margin-bottom:24px;border:2px solid}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px}
.metric{background:#0a0a0f;border:1px solid #1e1e2e;border-radius:8px;padding:16px;text-align:center}
.metric-val{font-size:1.4rem;font-weight:700;margin-bottom:4px}
.metric-lbl{font-size:0.7rem;color:#6b6b8a;letter-spacing:0.08em;text-transform:uppercase}
.frames-title{font-size:0.75rem;color:#6b6b8a;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px}
.frames-img{width:100%;border-radius:8px}
.spinner{display:none;text-align:center;padding:20px;color:#6b6b8a}
.footer{color:#6b6b8a;font-size:0.72rem;margin-top:32px;text-align:center}
</style>
</head>
<body>
<h1>Deep<span>Fake</span> Detector</h1>
<p class="sub">EfficientNet-B4 · Spatiotemporal Adapter · Transformer · Project 01</p>
<div class="card">
  <div class="upload-area" onclick="document.getElementById('fileInput').click()">
    <div class="upload-icon">🎬</div>
    <div class="upload-text"><strong>Click to upload a video</strong></div>
    <div class="upload-text">MP4, AVI, MOV, HEVC supported</div>
    <div class="file-name" id="fileName"></div>
  </div>
  <input type="file" id="fileInput" accept="video/*" onchange="onFileSelect(this)">
  <button class="btn" id="analyzeBtn" onclick="analyze()" disabled>Analyze Video</button>
</div>
<div class="spinner" id="spinner">⏳ Analyzing video... please wait</div>
<div class="result-card" id="resultCard">
  <div class="verdict" id="verdictBox"></div>
  <div class="metrics">
    <div class="metric"><div class="metric-val" id="fakePct">-</div><div class="metric-lbl">Fake Probability</div></div>
    <div class="metric"><div class="metric-val" id="confVal">-</div><div class="metric-lbl">Confidence</div></div>
    <div class="metric"><div class="metric-val" id="riskVal">-</div><div class="metric-lbl">Risk Level</div></div>
  </div>
  <div class="frames-title">Sampled Frames (8 frames analyzed)</div>
  <img class="frames-img" id="framesImg" src="" alt="frames">
</div>
<p class="footer">AI/ML Research · Digital Safety · Built with PyTorch + FastAPI</p>
<script>
let selectedFile=null;
function onFileSelect(input){
  if(input.files&&input.files[0]){
    selectedFile=input.files[0];
    document.getElementById('fileName').textContent=selectedFile.name;
    document.getElementById('analyzeBtn').disabled=false;
  }
}
async function analyze(){
  if(!selectedFile)return;
  document.getElementById('analyzeBtn').disabled=true;
  document.getElementById('spinner').style.display='block';
  document.getElementById('resultCard').style.display='none';
  const formData=new FormData();
  formData.append('file',selectedFile);
  try{
    const response=await fetch('/predict',{method:'POST',body:formData});
    const data=await response.json();
    if(data.error){alert('Error: '+data.error);}
    else{
      const vb=document.getElementById('verdictBox');
      vb.textContent=data.verdict;
      vb.style.color=data.color;
      vb.style.borderColor=data.color;
      document.getElementById('fakePct').textContent=data.fake_pct+'%';
      document.getElementById('confVal').textContent=data.confidence+'%';
      document.getElementById('riskVal').textContent=data.risk;
      document.getElementById('framesImg').src='data:image/png;base64,'+data.grid_b64;
      document.getElementById('resultCard').style.display='block';
    }
  }catch(e){alert('Request failed: '+e);}
  document.getElementById('spinner').style.display='none';
  document.getElementById('analyzeBtn').disabled=false;
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_PAGE

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = run_inference(tmp_path)
        os.unlink(tmp_path)
        if result is None:
            return JSONResponse({"error": "Could not read video file. Please try a different video."})
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
