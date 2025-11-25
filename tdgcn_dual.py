# tdgcn_dual_mirror.py
# - 처리(추론)도 미러 기준 가능 (PROC_MIRROR)
# - 표시 텍스트는 최종 디스플레이 프레임 좌상단 고정
# - handedness 자동 보정(auto): 시작 수초 관찰 후 Left/Right 스왑 여부 결정

import os, sys, time, yaml
import numpy as np
from collections import deque, Counter
import cv2, torch
import torch.nn as nn
import cv2, torch
import torch.nn as nn
# import mediapipe as mp # Moved to main

# ========= 사용자 설정 =========
TDGCN_REPO   = os.path.expanduser("./TD-GCN-Gesture")
CONFIG_YAML  = os.path.join(TDGCN_REPO, "config", "dhg14-28", "DHG14-28.yaml")
WEIGHTS_PATH = os.path.join(TDGCN_REPO, "checkpoints", "DHG", "DHG14label", "Sub3_j.pt")
SEQ_LEN = 64
PRINT_INTERVAL = 1.0
USE_GPU = torch.cuda.is_available()

# 처리/표시 미러 옵션
PROC_MIRROR = True     # 처리(추론) 입력을 좌우반전한 영상 기준으로 수행
DISP_MIRROR = True     # 표시도 미러(보통 PROC_MIRROR와 동일)
# handedness 보정 모드: 'auto' | 'swap' | 'none'
HANDEDNESS_MODE = 'auto'
# auto 보정 윈도우(초) 및 조건
AUTO_WINDOW_SEC = 1.5
AUTO_MIN_SAMPLES = 10  # 이 이상 샘플 모이면 판정

# ========= MediaPipe & 시각화 =========
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
MIN_DET_CONF, MIN_TRK_CONF = 0.5, 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ========= 유틸: MediaPipe 21 → DHG/SHREC 22 매핑 =========
def mediapipe21_to_dhg22(xyz21):
    wrist = xyz21[0]
    mcp_idx = [2, 5, 9, 13, 17]
    palm_center = (xyz21[[0] + mcp_idx].mean(axis=0))
    thumb  = np.stack([xyz21[2],  xyz21[3],  xyz21[4],  xyz21[1]], axis=0)
    indexf = np.stack([xyz21[5],  xyz21[6],  xyz21[7],  xyz21[8]],  axis=0)
    middle = np.stack([xyz21[9],  xyz21[10], xyz21[11], xyz21[12]], axis=0)
    ring   = np.stack([xyz21[13], xyz21[14], xyz21[15], xyz21[16]], axis=0)
    pinky  = np.stack([xyz21[17], xyz21[18], xyz21[19], xyz21[20]], axis=0)
    return np.concatenate([wrist[None,:], palm_center[None,:], thumb, indexf, middle, ring, pinky], axis=0)

def normalize_xyz(xyz, eps=1e-6):
    xyz = xyz.copy()
    xyz -= xyz[0]                         # wrist 원점 이동
    scale = np.linalg.norm(xyz[10]) + eps # middle MCP까지 거리
    xyz /= scale
    return xyz

# ========= TD-GCN 로더 =========
def build_tdgcn_and_load(weights_path, config_yaml, device):
    sys.path.append(TDGCN_REPO)
    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    model, last_err = None, None
    for mod_name, cls_name in [("model.tdgcn","Model"), ("model.model","Model"), ("model.tdgcn","TDGCN")]:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            ModelClass = getattr(mod, cls_name)
            model = ModelClass(**cfg.get("model_args", {}))
            break
        except Exception as e:
            last_err = e
    if model is None:
        raise RuntimeError(f"TD-GCN 모델 임포트 실패: {last_err}")

    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    feature_blob = {"feat": None}
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        print("[WARN] 마지막 Linear 레이어 미검출 → logits을 피처로 사용")
        return model, feature_blob, None

    def _hook(module, inputs):
        feature_blob["feat"] = inputs[0].detach().clone().cpu()

    last_linear.register_forward_pre_hook(_hook)
    return model, feature_blob, last_linear

def to_tdgcn_input(xyz_seq_22):
    x = torch.from_numpy(xyz_seq_22.astype(np.float32))  # (T,22,3)
    return x.permute(2,0,1).unsqueeze(0).unsqueeze(-1)   # (1,3,T,22,1)

def landmarks_to_world_xyz21(hand_landmarks, w, h):
    return np.array([(lm.x*w, lm.y*h, lm.z*max(w,h)) for lm in hand_landmarks.landmark], dtype=np.float32)

def put_hud_top_left(img, seq_buf, swap_on, proc_mirror):
    lines = [
        f"L: T={len(seq_buf['Left'])}/{SEQ_LEN}",
        f"R: T={len(seq_buf['Right'])}/{SEQ_LEN}",
        f"PROC_MIRROR={'ON' if proc_mirror else 'OFF'}  SWAP={'ON' if swap_on else 'OFF'}",
    ]
    x, y0 = 12, 28
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y0 + i*28), FONT, 0.8, (0,255,0), 2, cv2.LINE_AA)

def main():
    device = torch.device("cuda" if USE_GPU else "cpu")
    model_L, feat_blob_L, _ = build_tdgcn_and_load(WEIGHTS_PATH, CONFIG_YAML, device)
    model_R, feat_blob_R, _ = build_tdgcn_and_load(WEIGHTS_PATH, CONFIG_YAML, device)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=1,
        min_detection_confidence=MIN_DET_CONF, min_tracking_confidence=MIN_TRK_CONF
    )

    seq_buf   = {"Left": deque(maxlen=SEQ_LEN), "Right": deque(maxlen=SEQ_LEN)}
    last_print= {"Left": 0.0, "Right": 0.0}

    # ==== handedness 자동 보정 상태 ====
    start_time   = time.time()
    auto_samples = []        # [(label, cx_norm)] — 한 손만 보이는 프레임만 수집
    auto_swap    = False

    # 전역 설정의 로컬 복사본 (전역을 직접 수정하지 않음)
    handedness_mode = HANDEDNESS_MODE

    # base swap: PROC_MIRROR일 때는 한 번 스왑 (미러 처리 보정)
    base_swap = PROC_MIRROR
    # user mode swap
    mode_swap = (handedness_mode == 'swap')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("카메라 프레임을 읽을 수 없습니다.")
                break

            # 처리용 프레임 (미러 적용)
            proc = cv2.flip(frame, 1) if PROC_MIRROR else frame
            h, w = proc.shape[:2]

            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            hands_list = []
            if results.multi_hand_landmarks and results.multi_handedness:
                for i, hl in enumerate(results.multi_hand_landmarks):
                    # 랜드마크 그리기(처리 좌표계)
                    mp_draw.draw_landmarks(
                        vis, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    # handedness 최고 score
                    best = max(results.multi_handedness[i].classification, key=lambda c: c.score)
                    label = best.label  # 'Left' or 'Right'
                    # 손 중심 x좌표(정규화) — 0(좌) ~ 1(우)
                    cx = np.mean([p.x for p in hl.landmark])
                    hands_list.append((label, hl, cx))

            # ==== AUTO 보정 수집 (한 손만 보일 때만) ====
            if handedness_mode == 'auto':
                if len(hands_list) == 1:
                    label, _, cx = hands_list[0]
                    auto_samples.append((label, cx))
                # 일정 시간 또는 샘플 수가 모이면 판정
                if (time.time() - start_time > AUTO_WINDOW_SEC) and (len(auto_samples) >= AUTO_MIN_SAMPLES):
                    right_half = [(lab, cx) for (lab, cx) in auto_samples if cx > 0.5]
                    left_half  = [(lab, cx) for (lab, cx) in auto_samples if cx <= 0.5]
                    right_counts = Counter([lab for (lab, _) in right_half])
                    left_counts  = Counter([lab for (lab, _) in left_half])
                    # 판정 규칙: 오른쪽 절반에 Left가 많고 / 왼쪽 절반에 Right가 많으면 swap 필요
                    swap_score = right_counts.get('Left', 0) + left_counts.get('Right', 0)
                    keep_score = right_counts.get('Right', 0) + left_counts.get('Left', 0)
                    auto_swap = (swap_score > keep_score)
                    handedness_mode = 'none'  # 확정
                    print(f"[AUTO] samples={len(auto_samples)} swap={auto_swap} (swap={swap_score}, keep={keep_score})")

            # 최종 스왑 여부
            mode_swap = (handedness_mode == 'swap')
            effective_swap = base_swap ^ mode_swap ^ auto_swap

            # ==== 버퍼 적재 ====
            for label, hl, cx in hands_list:
                eff_label = ('Left' if label == 'Right' else 'Right') if effective_swap else label
                if eff_label not in ('Left', 'Right'):
                    continue
                xyz21 = landmarks_to_world_xyz21(hl, w, h)
                xyz22 = mediapipe21_to_dhg22(xyz21)
                xyz22 = normalize_xyz(xyz22)
                seq_buf[eff_label].append(xyz22)

            # ==== 1초 간격 피처 추출 ====
            now = time.time()
            if len(seq_buf["Right"]) == SEQ_LEN and now - last_print["Right"] >= PRINT_INTERVAL:
                last_print["Right"] = now
                x = to_tdgcn_input(np.stack(list(seq_buf["Right"]), axis=0)).to(device)
                with torch.no_grad():
                    logits_R = model_R(x)
                    feat_R = (feat_blob_R["feat"].cpu().numpy().squeeze(0)
                              if feat_blob_R["feat"] is not None else
                              logits_R.cpu().numpy().squeeze(0))
                print(f"=== [Right] TD-GCN Feature @ {now:.1f}s ===")
                print(" seq_len:", len(seq_buf["Right"]), "| feature_dim:", feat_R.shape[0])
                print(" feature[:10]:", np.array2string(feat_R[:10], precision=4, separator=', '))

            if len(seq_buf["Left"]) == SEQ_LEN and now - last_print["Left"] >= PRINT_INTERVAL:
                last_print["Left"] = now
                x = to_tdgcn_input(np.stack(list(seq_buf["Left"]), axis=0)).to(device)
                with torch.no_grad():
                    logits_L = model_L(x)
                    feat_L = (feat_blob_L["feat"].cpu().numpy().squeeze(0)
                              if feat_blob_L["feat"] is not None else
                              logits_L.cpu().numpy().squeeze(0))
                print(f"=== [Left] TD-GCN Feature @ {now:.1f}s ===")
                print(" seq_len:", len(seq_buf["Left"]), "| feature_dim:", feat_L.shape[0])
                print(" feature[:10]:", np.array2string(feat_L[:10], precision=4, separator=', '))

            # ==== 최종 표시 프레임 ====
            disp = proc.copy()  # proc에 이미 미러 반영됨(=처리 좌표계와 동일)
            if DISP_MIRROR and not PROC_MIRROR:
                disp = cv2.flip(disp, 1)  # 처리와 표시를 분리하고 싶을 때만
            put_hud_top_left(disp, seq_buf, swap_on=effective_swap, proc_mirror=PROC_MIRROR)

            title = "Hands + Dual TD-GCN (proc=mirror)" if PROC_MIRROR else "Hands + Dual TD-GCN (proc=normal)"
            cv2.imshow(title, disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

class TDGCN_Dual_Encoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model, self.feature_blob, _ = build_tdgcn_and_load(WEIGHTS_PATH, CONFIG_YAML, device)
        # [Efficiency] Remove classifier
        self.model.fc = nn.Identity()
        self.device = device

    def forward(self, x):
        # x: (B, C, T, J, 1)
        # TDGCN expects (B, C, T, J, 1) or similar. 
        # The original code uses to_tdgcn_input which produces (1,3,T,22,1).
        # Here we assume batch input.
        logits = self.model(x)
        if self.feature_blob["feat"] is not None:
            return self.feature_blob["feat"]
        return logits

def get_encoder(device):
    return TDGCN_Dual_Encoder(device)

if __name__ == "__main__":
    main()