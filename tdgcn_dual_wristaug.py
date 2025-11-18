#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD-GCN Dual Inference (Mirror + AUTO handedness) with Wrist-World as AUXILIARY embedding (C stays 3)
-----------------------------------------------------------------------------------------------
Context:
- Available checkpoint: C=3 (xyz only), single-hand dataset.
- No checkpoint trained with wrist-world in input.

Goal:
- Keep the encoder input as (C=3) to stay compatible with the existing checkpoint.
- Still leverage wrist world information by concatenating a compact summary vector of the
  wrist trajectory to the extracted encoder embedding (late-fusion at embedding-level).

What this script does:
- Keeps original TD-GCN input: (1,3,T,22,1)
- Extracts encoder embedding by hooking the last Linear (pre-hook). If not found, uses logits.
- Computes per-sequence wrist-world normalization: (x/w, y/h, z/max(w,h)) per frame.
- Builds an AUX vector: concat(mean, std) over time for wrist-world → shape (6,).
- Final embedding: z_cat = concat([z_encoder, aux_wrist])  # dimension: D + 6
- Optionally saves embeddings to NPY.
- 좌표계: 왼손 손목(world)을 전역 원점으로 사용하고, 모든 손 좌표를 그 origin 기준 상대 좌표로 변환.
- 화면 좌측 상단 HUD에 왼손/오른손 wrist가 왼손에서 얼마나 떨어져 있는지(norm) 표시.

Note:
- This is late-fusion; classifier/fine-tuning should expect the augmented dim (D+6) if used.
- For 2-hand streaming, we run the same model twice (Left/Right) with M=1.

Usage:
  python tdgcn_dual_mirror_wristaux.py \
    --save-emb out_dir \
    --dump-every 10

Press 'q' to quit.
"""

import os, sys, time, yaml, argparse
import numpy as np
from collections import deque, Counter
import cv2, torch
import torch.nn as nn
import mediapipe as mp

# ========= 사용자 기본 설정 (필요시 인자 override) =========
TDGCN_REPO   = os.path.expanduser("./TD-GCN-Gesture")
CONFIG_YAML  = os.path.join(TDGCN_REPO, "config", "dhg14-28", "DHG14-28.yaml")
WEIGHTS_PATH = os.path.join(TDGCN_REPO, "checkpoints", "DHG", "DHG14label", "Sub3_j.pt")
SEQ_LEN = 64
PRINT_INTERVAL = 1.0

# 미러 & 핸디드니스 옵션
PROC_MIRROR = True
DISP_MIRROR = True
HANDEDNESS_MODE = 'auto'  # 'auto' | 'swap' | 'none'
AUTO_WINDOW_SEC = 1.5
AUTO_MIN_SAMPLES = 10

# 카메라/시각화
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
MIN_DET_CONF, MIN_TRK_CONF = 0.5, 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ========= 유틸: MediaPipe 21 → DHG/SHREC 22 매핑 & 정규화 =========

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


def normalize_xyz(xyz, origin=None, eps=1e-6):
    """
    xyz: (22,3) 혹은 (N,3)
    origin: 기준 원점 (3,) – None이면 첫 관절(0번, wrist)을 origin으로 사용
    """
    xyz = xyz.copy()
    if origin is None:
        origin = xyz[0]
    xyz -= origin                         # 지정한 원점 기준으로 평행이동
    scale = np.linalg.norm(xyz[10]) + eps # middle MCP까지 거리로 스케일 정규화
    xyz /= scale
    return xyz


def to_tdgcn_input(xyz_seq_22):
    x = torch.from_numpy(xyz_seq_22.astype(np.float32))  # (T,22,3)
    return x.permute(2,0,1).unsqueeze(0).unsqueeze(-1)   # (1,3,T,22,1)


def landmarks_to_world_xyz21(hand_landmarks, w, h):
    # pixel-like world coords; we'll normalize per-frame to [0,1] scale later
    return np.array([(lm.x*w, lm.y*h, lm.z*max(w,h)) for lm in hand_landmarks.landmark], dtype=np.float32)


def wrist_world_norm(wrist_world_xyz, w, h):
    """Per-frame normalization of wrist world coords to [0,1]-ish scale.
    Args:
      wrist_world_xyz: (3,) in pixels
    Returns:
      (3,) normalized as (x/w, y/h, z/max(w,h))
    """
    return np.array([
        wrist_world_xyz[0] / max(w, 1e-6),
        wrist_world_xyz[1] / max(h, 1e-6),
        wrist_world_xyz[2] / max(max(w,h), 1e-6)
    ], dtype=np.float32)

# ========= TD-GCN 로더 (안전 로더 + hook) =========

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

    # 안전 로드: PyTorch 경고 회피 + 호환되지 않는 키 무시
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] missing keys:", list(missing)[:8], '...')
    if unexpected:
        print("[WARN] unexpected keys:", list(unexpected)[:8], '...')

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
        feature_blob["feat"] = inputs[0].detach().clone().cpu()  # (N, D)
    last_linear.register_forward_pre_hook(_hook)
    return model, feature_blob, last_linear

# ========= HUD =========

def put_hud_top_left(img, seq_buf, swap_on, proc_mirror, aux_dim, left_dist, right_dist):
    lines = [
        f"L: T={len(seq_buf['Left'])}/{SEQ_LEN}  dist={left_dist:.3f}",
        f"R: T={len(seq_buf['Right'])}/{SEQ_LEN} dist={right_dist:.3f}",
        f"PROC_MIRROR={'ON' if proc_mirror else 'OFF'}  SWAP={'ON' if swap_on else 'OFF'}",
        f"ENC C=3  | AUX(wrist) dim={aux_dim}"
    ]
    x, y0 = 12, 28
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y0 + i*28), FONT, 0.8, (0,255,0), 2, cv2.LINE_AA)

# ========= AUX 벡터 생성 =========

def build_wrist_aux(seq_wrist_norm: np.ndarray) -> np.ndarray:
    """seq_wrist_norm: (T,3) normalized wrist coords over time
    Returns aux: concat(mean(3), std(3)) -> (6,)
    """
    if seq_wrist_norm.shape[0] == 0:
        return np.zeros((6,), dtype=np.float32)
    mu = seq_wrist_norm.mean(axis=0)
    sd = seq_wrist_norm.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)

# ========= 메인 =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=CAMERA_INDEX)
    ap.add_argument('--proc-mirror', action='store_true', default=PROC_MIRROR)
    ap.add_argument('--disp-mirror', action='store_true', default=DISP_MIRROR)
    ap.add_argument('--seq-len', type=int, default=SEQ_LEN)
    ap.add_argument('--save-emb', type=str, default='')
    ap.add_argument('--dump-every', type=int, default=0, help='save every N prints (0=off)')
    ap.add_argument('--use-gpu', action='store_true', default=torch.cuda.is_available())
    args = ap.parse_args()

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    model_L, feat_L, _ = build_tdgcn_and_load(WEIGHTS_PATH, CONFIG_YAML, device)
    model_R, feat_R, _ = build_tdgcn_and_load(WEIGHTS_PATH, CONFIG_YAML, device)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=1,
        min_detection_confidence=MIN_DET_CONF, min_tracking_confidence=MIN_TRK_CONF
    )

    SEQ = args.seq_len
    seq_buf   = {"Left": deque(maxlen=SEQ), "Right": deque(maxlen=SEQ)}          # (T,22,3) normalized xyz
    wrist_buf = {"Left": deque(maxlen=SEQ), "Right": deque(maxlen=SEQ)}          # (T,3) normalized wrist world
    last_print= {"Left": 0.0, "Right": 0.0}
    print_count= {"Left": 0, "Right": 0}

    start_time   = time.time()
    auto_samples = []
    auto_swap    = False
    handedness_mode = HANDEDNESS_MODE
    base_swap = args.proc_mirror
    mode_swap = (handedness_mode == 'swap')

    # 왼손 손목 world 좌표 (전역 기준 원점)
    left_wrist_world = None

    # 상대 거리 HUD용 변수
    left_dist = 0.0
    right_dist = 0.0

    os.makedirs(args.save_emb, exist_ok=True) if args.save_emb else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('카메라 프레임을 읽을 수 없습니다.'); break

            proc = cv2.flip(frame, 1) if args.proc_mirror else frame
            h, w = proc.shape[:2]

            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            hands_list = []
            if results.multi_hand_landmarks and results.multi_handedness:
                for i, hl in enumerate(results.multi_hand_landmarks):
                    mp_draw.draw_landmarks(
                        vis, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                    best = max(results.multi_handedness[i].classification, key=lambda c: c.score)
                    label = best.label
                    cx = np.mean([p.x for p in hl.landmark])
                    hands_list.append((label, hl, cx))

            # AUTO swap (한 손만 보일 때만 샘플 수집)
            if handedness_mode == 'auto':
                if len(hands_list) == 1:
                    label, _, cx = hands_list[0]
                    auto_samples.append((label, cx))
                if (time.time() - start_time > AUTO_WINDOW_SEC) and (len(auto_samples) >= AUTO_MIN_SAMPLES):
                    right_half = [(lab, cx) for (lab, cx) in auto_samples if cx > 0.5]
                    left_half  = [(lab, cx) for (lab, cx) in auto_samples if cx <= 0.5]
                    right_counts = Counter([lab for (lab, _) in right_half])
                    left_counts  = Counter([lab for (lab, _) in left_half])
                    swap_score = right_counts.get('Left', 0) + left_counts.get('Right', 0)
                    keep_score = right_counts.get('Right', 0) + left_counts.get('Left', 0)
                    auto_swap = (swap_score > keep_score)
                    handedness_mode = 'none'
                    print(f"[AUTO] samples={len(auto_samples)} swap={auto_swap} (swap={swap_score}, keep={keep_score})")

            mode_swap = (handedness_mode == 'swap')
            effective_swap = base_swap ^ mode_swap ^ auto_swap

            # 버퍼 적재
            for label, hl, cx in hands_list:
                eff_label = ('Left' if label == 'Right' else 'Right') if effective_swap else label
                if eff_label not in ('Left', 'Right'):
                    continue

                # 1) MediaPipe world 좌표 (px 단위)
                xyz21_world = landmarks_to_world_xyz21(hl, w, h)

                # 2) DHG 22-joint world 좌표로 변환
                xyz22_world = mediapipe21_to_dhg22(xyz21_world)  # (22,3), 아직 원점/스케일 정규화 전

                # 3) 왼손 손목 world 좌표를 전역 기준으로 저장 (왼손이 보이는 순간 업데이트)
                if eff_label == 'Left':
                    left_wrist_world = xyz22_world[0].copy()

                # 4) 이번 프레임에서 사용할 정규화 origin 결정
                #    - 왼손을 한 번이라도 본 이후에는 항상 왼손 손목을 원점으로 사용
                #    - 아직 왼손을 본 적 없으면 해당 손의 wrist를 임시 원점으로 사용
                if left_wrist_world is not None:
                    origin = left_wrist_world
                else:
                    origin = xyz22_world[0]

                # 5) origin 기준으로 평행이동 + 스케일 정규화 → TD-GCN 입력
                xyz22 = normalize_xyz(xyz22_world, origin=origin)  # 왼손 손목을 (0,0,0)으로
                seq_buf[eff_label].append(xyz22)

                # 오른손도 "왼손에서 얼마나 떨어져 있는지" 직접 확인할 수 있도록 디버그 출력
                # (왼손 wrist 기준 상대 좌표에서 해당 손의 wrist 좌표)
                print(f"[DEBUG] {eff_label} wrist (relative to left): {xyz22[0]}")

                # 상대 거리(norm) 계산해서 HUD용으로 저장
                dist = float(np.linalg.norm(xyz22[0]))
                if eff_label == "Left":
                    left_dist = dist
                else:
                    right_dist = dist

                # 6) AUX용 wrist world (원래 정의 유지: world→(x/w,y/h,z/max(w,h)))
                wristW = xyz22_world[0]
                wristN = wrist_world_norm(wristW, w, h)  # (3,)
                wrist_buf[eff_label].append(wristN)

            # 1초 간격 임베딩/로짓 출력
            now = time.time()
            for side, model, feat_blob in [("Right", model_R, feat_R), ("Left", model_L, feat_L)]:
                if len(seq_buf[side]) == SEQ and now - last_print[side] >= PRINT_INTERVAL:
                    last_print[side] = now
                    print_count[side] += 1

                    arrN = np.stack(list(seq_buf[side]), axis=0)    # (T,22,3)
                    x = to_tdgcn_input(arrN).to(device)             # (1,3,T,22,1)
                    with torch.no_grad():
                        logits = model(x)
                        enc = (feat_blob["feat"].cpu().numpy().squeeze(0)
                               if feat_blob["feat"] is not None else
                               logits.cpu().numpy().squeeze(0))
                    # AUX wrist summary (mean+std)
                    wrN = np.stack(list(wrist_buf[side]), axis=0) if len(wrist_buf[side])>0 else np.zeros((0,3), np.float32)
                    aux = build_wrist_aux(wrN)  # (6,)
                    z_cat = np.concatenate([enc, aux], axis=0)

                    print(f"=== [{side}] TD-GCN + AUX(wrist) @ {now:.1f}s ===")
                    print(" input:", tuple(x.shape), " logits:", tuple(logits.shape), " enc_dim:", enc.shape[0], " aux_dim:", aux.shape[0], " z_cat:", z_cat.shape[0])
                    print(" z_cat[:10]:", np.array2string(z_cat[:10], precision=4, separator=', '))

                    # Optional save
                    if args.save_emb and (args.dump_every>0) and (print_count[side] % args.dump_every == 0):
                        ts = int(now)
                        np.save(os.path.join(args.save_emb, f"{side}_enc_{ts}.npy"), enc)
                        np.save(os.path.join(args.save_emb, f"{side}_aux_{ts}.npy"), aux)
                        np.save(os.path.join(args.save_emb, f"{side}_zcat_{ts}.npy"), z_cat)

            # 디스플레이
            disp = proc.copy()
            if args.disp_mirror and not args.proc_mirror:
                disp = cv2.flip(disp, 1)
            put_hud_top_left(
                disp,
                seq_buf,
                swap_on=effective_swap,
                proc_mirror=args.proc_mirror,
                aux_dim=6,
                left_dist=left_dist,
                right_dist=right_dist
            )

            title = "Hands + TD-GCN (enc C=3, AUX wrist, Left-wrist-relative)"
            cv2.imshow(title, disp)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()