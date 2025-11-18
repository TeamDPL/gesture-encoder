# stgcn_test.py — Realtime single-hand ST-GCN-SL (MediaPipe→V adapt for ASLLVD weights)
import os
import sys
import time
import yaml
import importlib
import numpy as np
from pathlib import Path
from collections import deque
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
import types

# ================== 경로/설정 ==================
SCRIPT_DIR     = Path(__file__).resolve().parent
STGCN_SL_REPO  = (SCRIPT_DIR / "st-gcn-sl").resolve()        # 깃 루트
STGCN_SL_ROOT  = (STGCN_SL_REPO / "st-gcn").resolve()        # 코드 루트
TORCHLIGHT_DIR = (STGCN_SL_ROOT / "torchlight").resolve()    # torchlight 패키지

# ▶ ASLLVD-Skeleton config & weights 
CONFIG_YAML = STGCN_SL_ROOT / "config" / "sl" / "train-asllvd-skeleton.yaml"
WEIGHTS_PATH = STGCN_SL_REPO / "pre-trained/epoch1350_model.pt"          # 네가 받아 둔 파일

# sys.path 주입
for p in (STGCN_SL_ROOT, TORCHLIGHT_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

# 필수 파일 체크
assert (STGCN_SL_ROOT / "net" / "st_gcn.py").exists(), "st-gcn/net/st_gcn.py not found"
assert Path(CONFIG_YAML).exists(), f"CONFIG_YAML not found: {CONFIG_YAML}"
assert Path(WEIGHTS_PATH).exists(), f"WEIGHTS_PATH not found: {WEIGHTS_PATH}"

# 실시간 하이퍼파라미터
SEQ_LEN         = 64
PRINT_INTERVAL  = 1.0
USE_GPU         = torch.cuda.is_available()
CAMERA_INDEX    = 0
FRAME_W, FRAME_H = 1280, 720
MIN_DET_CONF, MIN_TRK_CONF = 0.5, 0.5

# ================== skvideo 스텁(유틸 의존성 우회) ==================
def _stub_skvideo():
    try:
        import skvideo.io  # noqa: F401
        return
    except Exception:
        skvideo_mod = types.ModuleType("skvideo")
        io_mod = types.ModuleType("io")
        def _not_impl(*args, **kwargs):
            raise RuntimeError("skvideo is stubbed; install scikit-video if you need video IO.")
        io_mod.vread = _not_impl
        io_mod.vwrite = _not_impl
        sys.modules["skvideo"] = skvideo_mod
        sys.modules["skvideo.io"] = io_mod
_stub_skvideo()

# ================== MediaPipe 준비 ==================
mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ================== 유틸 (단손) ==================
def extract_xyz21(landmarks, w, h):
    """MediaPipe 21점 -> (21,3)"""
    if landmarks is None:
        return np.zeros((21,3), dtype=np.float32)
    pts = []
    for lm in landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        z = lm.z * max(w, h)
        pts.append((x, y, z))
    return np.asarray(pts, dtype=np.float32)

def pick_single_hand(results):
    """오른손 우선, 없으면 왼손 하나만 선택"""
    if not results.multi_hand_landmarks:
        return None
    # label 매칭
    chosen = None
    if results.multi_handedness:
        for i, h in enumerate(results.multi_handedness):
            if h.classification[0].label == 'Right':
                chosen = results.multi_hand_landmarks[i]
                break
    if chosen is None:
        chosen = results.multi_hand_landmarks[0]
    return chosen

def normalize_xyz21_single(xyz21, eps=1e-6):
    """단손 정규화: 손목 원점, 중지 MCP(9) 기준 스케일"""
    xyz = xyz21.copy()
    wrist = xyz[0]
    xyz -= wrist
    ref = xyz[9]
    s = np.linalg.norm(ref) + eps
    xyz /= s
    return xyz

def to_stgcn_input(xyz_seq_V3):
    """(T,V,3) -> (1,3,T,V,1)"""
    x = torch.from_numpy(xyz_seq_V3.astype(np.float32))
    x = x.permute(2,0,1).unsqueeze(0).unsqueeze(-1)
    return x

# ================== 모델 유틸 ==================
def infer_expected_V(model, cfg):
    """data_bn.num_features = in_channels * V * num_person → V 추정"""
    in_channels = int(cfg.get("model_args", {}).get("in_channels", 3))
    num_person  = int(cfg.get("model_args", {}).get("num_person", 1))
    if hasattr(model, "data_bn"):
        nf = int(model.data_bn.num_features)
        V = nf // (in_channels * num_person)
        return V
    return int(cfg.get("model_args", {}).get("num_point", 42))

def build_frame_xyz_single(results, w, h, expected_V):
    """
    기대 V에 맞춰 (V,3) 프레임 구성:
      - 손 1개(오른손 우선) 21점 사용 후,
      - expected_V > 21 이면 (expected_V - 21) zero padding
      - expected_V < 21 이면 앞 expected_V개만 사용(권장X, 임시)
    """
    hl = pick_single_hand(results)
    xyz21 = extract_xyz21(hl, w, h)
    xyz21 = normalize_xyz21_single(xyz21)

    if expected_V == 21:
        return xyz21
    elif expected_V > 21:
        pad = np.zeros((expected_V - 21, 3), dtype=np.float32)
        return np.concatenate([xyz21, pad], axis=0)
    else:
        # 임시: 앞 expected_V개만 사용
        return xyz21[:expected_V, :]

# ================== ST-GCN-SL 로더 ==================
def build_stgcn_sl_and_load(weights_path, config_yaml, device):
    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    # net.st_gcn 임포트
    mod = importlib.import_module("net.st_gcn")

    # 클래스 선택
    ModelClass = None
    for name in ("Model", "STGCN"):
        if hasattr(mod, name):
            ModelClass = getattr(mod, name)
            break
    if ModelClass is None:
        avail = [n for n, v in mod.__dict__.items() if isinstance(v, type)]
        raise RuntimeError(f"net.st_gcn에 Model/STGCN 없음. found={avail}")

    # 모델 생성
    model = ModelClass(**cfg.get("model_args", {}))

    # 가중치 로드 (torch>=2.4 대비)
    try:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=True)  # type: ignore
    except TypeError:
        ckpt = torch.load(str(weights_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[WARN] state_dict mismatch → missing:", missing, "unexpected:", unexpected)

    model.eval().to(device)
    return model, cfg

# ================== 메인 루프 ==================
def main():
    device = torch.device("cuda" if USE_GPU else "cpu")
    model, cfg = build_stgcn_sl_and_load(WEIGHTS_PATH, CONFIG_YAML, device)

    expected_V = infer_expected_V(model, cfg)
    print(f"[INFO] Model expects V={expected_V} nodes (single-hand feed)")

    label_map = cfg.get("label_map", None)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=1,
        min_detection_confidence=MIN_DET_CONF, min_tracking_confidence=MIN_TRK_CONF
    )

    # 분류기 직전 feature hook (있으면 사용)
    feature_blob = {"feat": None}
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        print("[WARN] 마지막 Linear 미발견 → logits을 임시 임베딩으로 사용")
    else:
        def _hook(module, inputs):
            feature_blob["feat"] = inputs[0].detach().clone().cpu()
        last_linear.register_forward_pre_hook(_hook)

    seq_buf = deque(maxlen=SEQ_LEN)
    last_print = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # 시각화
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        out, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # 단손 프레임 생성 (V에 맞춰 패딩/절단)
            xyz = build_frame_xyz_single(results, w, h, expected_V)  # (V,3)
            seq_buf.append(xyz)

            # 주기적 추론
            now = time.time()
            if now - last_print >= PRINT_INTERVAL and len(seq_buf) == SEQ_LEN:
                last_print = now
                seq_np = np.stack(list(seq_buf), axis=0)          # (T,V,3)
                x = to_stgcn_input(seq_np).to(device)             # (1,3,T,V,1)

                with torch.no_grad():
                    logits = model(x)                             # (1, num_classes)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    top1 = int(np.argmax(probs))
                    conf = float(probs[top1])
                    if feature_blob["feat"] is not None:
                        feat = feature_blob["feat"].cpu().numpy().squeeze(0)
                    else:
                        feat = logits.cpu().numpy().squeeze(0)

                cls_name = str(top1)
                if isinstance(label_map, dict):
                    cls_name = label_map.get(top1, label_map.get(str(top1), str(top1)))

                print("=== ST-GCN-SL (Single-hand) ===")
                print(f" T={len(seq_buf)} | top1: {cls_name}  (p={conf:.3f})")
                print(" feature_dim:", feat.shape[0] if hasattr(feat, 'shape') else len(feat))
                print(" feature[:10]:", np.array2string(np.asarray(feat)[:10], precision=4, separator=', '))

            # OSD
            cv2.putText(out, f"T={len(seq_buf)}/{SEQ_LEN}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("ST-GCN-SL Single-hand (press 'q' to quit)", out)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()