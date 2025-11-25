# stgcn_1_test.py — ST-GCN-SL, single-hand, T=1 (no buffer)
import os, sys, time, yaml, importlib, types
import numpy as np
from pathlib import Path
import cv2, torch
import torch.nn as nn
import cv2, torch
import torch.nn as nn
# import mediapipe as mp

# ===== Paths =====
SCRIPT_DIR     = Path(__file__).resolve().parent
STGCN_SL_REPO  = (SCRIPT_DIR / "st-gcn-sl").resolve()
STGCN_SL_ROOT  = (STGCN_SL_REPO / "st-gcn").resolve()
TORCHLIGHT_DIR = (STGCN_SL_ROOT / "torchlight").resolve()

CONFIG_YAML = STGCN_SL_ROOT / "config" / "sl" / "train-asllvd-skeleton.yaml"
WEIGHTS_PATH = STGCN_SL_REPO / "pre-trained/epoch1350_model.pt"

for p in (STGCN_SL_ROOT, TORCHLIGHT_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

assert (STGCN_SL_ROOT / "net" / "st_gcn.py").exists()
assert CONFIG_YAML.exists(), f"Missing {CONFIG_YAML}"
assert WEIGHTS_PATH.exists(), f"Missing {WEIGHTS_PATH}"

# ===== Runtime =====
USE_GPU = torch.cuda.is_available()
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
MIN_DET_CONF, MIN_TRK_CONF = 0.5, 0.5

# ===== skvideo stub to avoid optional dep =====
def _stub_skvideo():
    try:
        import skvideo.io  # noqa
    except Exception:
        skvideo_mod = types.ModuleType("skvideo")
        io_mod = types.ModuleType("io")
        def _not_impl(*a, **k): raise RuntimeError("skvideo stubbed.")
        io_mod.vread = _not_impl; io_mod.vwrite = _not_impl
        sys.modules["skvideo"] = skvideo_mod; sys.modules["skvideo.io"] = io_mod
_stub_skvideo()

# ===== MediaPipe =====
# mp_hands  = mp.solutions.hands
# mp_draw   = mp.solutions.drawing_utils
# mp_styles = mp.solutions.drawing_styles

# ===== helpers =====
def extract_xyz21(hl, w, h):
    if hl is None: return np.zeros((21,3), np.float32)
    pts = [(lm.x*w, lm.y*h, lm.z*max(w,h)) for lm in hl.landmark]
    return np.asarray(pts, np.float32)

def pick_single_hand(results):
    if not results.multi_hand_landmarks: return None
    if results.multi_handedness:
        for i, h in enumerate(results.multi_handedness):
            if h.classification[0].label == 'Right':
                return results.multi_hand_landmarks[i]
    return results.multi_hand_landmarks[0]

def normalize_xyz21(xyz, eps=1e-6):
    xyz = xyz.copy()
    xyz -= xyz[0]              # wrist origin
    s = np.linalg.norm(xyz[9]) + eps   # middle MCP scale
    xyz /= s
    return xyz

def to_tensor_T1(xyz_V3):
    # (V,3) -> (1,3,1,V,1)
    x = torch.from_numpy(xyz_V3.astype(np.float32))
    x = x.permute(1,0).unsqueeze(0).unsqueeze(2).unsqueeze(-1)
    return x

def infer_expected_V(model, cfg):
    inc = int(cfg.get("model_args", {}).get("in_channels", 3))
    M   = int(cfg.get("model_args", {}).get("num_person", 1))
    if hasattr(model, "data_bn"):
        nf = int(model.data_bn.num_features)
        return nf // (inc * M)
    return int(cfg.get("model_args", {}).get("num_point", 21))

# ===== load model =====
with open(CONFIG_YAML, "r") as f:
    cfg = yaml.safe_load(f)
mod = importlib.import_module("net.st_gcn")
ModelClass = getattr(mod, "Model", None) or getattr(mod, "STGCN")
model = ModelClass(**cfg.get("model_args", {}))
try:
    ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)  # type: ignore
except TypeError:
    ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu")
state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    print("[WARN] state_dict mismatch:", "missing:", missing, "unexpected:", unexpected)
model.eval().to(torch.device("cuda" if USE_GPU else "cpu"))

expected_V = infer_expected_V(model, cfg)
print(f"[INFO] Model expects V={expected_V} (single-frame inference)")

# feature hook (optional)
feature_blob = {"feat": None}
last_fc = None
for m in model.modules():
    if isinstance(m, nn.Linear): last_fc = m
if last_fc:
    def _hook(module, inputs): feature_blob["feat"] = inputs[0].detach().clone().cpu()
    last_fc.register_forward_pre_hook(_hook)
else:
    print("[WARN] No Linear head found → using logits as embedding")

label_map = cfg.get("label_map", None)

def main():
    # ===== run =====
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    import mediapipe as mp
    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=1,
        min_detection_confidence=MIN_DET_CONF, min_tracking_confidence=MIN_TRK_CONF
    )

    device = torch.device("cuda" if USE_GPU else "cpu")
    prev_t = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # draw
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        out, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # single-hand frame → adapt to expected_V
            hl = pick_single_hand(results)
            xyz21 = normalize_xyz21(extract_xyz21(hl, w, h))
            if expected_V > 21:
                pad = np.zeros((expected_V-21, 3), np.float32)
                xyzV = np.concatenate([xyz21, pad], axis=0)
            else:
                xyzV = xyz21[:expected_V, :]

            # infer (T=1)
            x = to_tensor_T1(xyzV).to(device)           # (1,3,1,V,1)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            top1 = int(np.argmax(probs))
            conf = float(probs[top1])
            cls_name = str(top1)
            if isinstance(label_map, dict):
                cls_name = label_map.get(top1, label_map.get(str(top1), str(top1)))

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_t + 1e-9)
            prev_t = now

            # overlay
            cv2.putText(out, f"{cls_name}  p={conf:.2f}  FPS={fps:.1f}", (12, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(out, f"V={expected_V}, T=1", (12, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

            cv2.imshow("ST-GCN-SL Single-frame (q to quit)", out)
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
    finally:
        hands.close(); cap.release(); cv2.destroyAllWindows()

class STGCN_SL_1_Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load model logic copied from script body
        with open(CONFIG_YAML, "r") as f:
            cfg = yaml.safe_load(f)
        mod = importlib.import_module("net.st_gcn")
        ModelClass = getattr(mod, "Model", None) or getattr(mod, "STGCN")
        self.model = ModelClass(**cfg.get("model_args", {}))
        try:
            ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state, strict=False)
        self.model.eval().to(device)
        self.device = device

        self.feature_blob = {"feat": None}
        last_fc = None
        for m in self.model.modules():
            if isinstance(m, nn.Linear): last_fc = m
        if last_fc:
            def _hook(module, inputs): self.feature_blob["feat"] = inputs[0]
            last_fc.register_forward_pre_hook(_hook)
            
        # [Efficiency] Remove classifier
        self.model.fcn = nn.Identity()

    def forward(self, x):
        # x: (B, 3, 1, V, 1)
        logits = self.model(x)
        if self.feature_blob["feat"] is not None:
            return self.feature_blob["feat"]
        return logits

def get_encoder(device):
    return STGCN_SL_1_Encoder(device)

if __name__ == "__main__":
    main()