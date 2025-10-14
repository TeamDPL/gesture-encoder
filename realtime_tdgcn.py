# realtime_tdgcn.py
import os
import sys
import time
import yaml
import math
import numpy as np
from collections import deque 
import cv2
import torch
import torch.nn as nn
import mediapipe as mp

# ========= 사용자 설정 =========
TDGCN_REPO = os.path.expanduser("./TD-GCN-Gesture")  # 클론한 경로
CONFIG_YAML = os.path.join(TDGCN_REPO, "config", "dhg14-28", "DHG14-28.yaml")  # 사용할 설정
WEIGHTS_PATH = os.path.join(TDGCN_REPO, "checkpoints", "DHG", "DHG14label", "Sub3_j.pt")  # 체크포인트 경로
SEQ_LEN = 64            # TD-GCN 입력 길이(T)
PRINT_INTERVAL = 1.0    # 초(피처 출력 주기)
USE_GPU = torch.cuda.is_available() 

# ========= MediaPipe & 시각화 =========
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
MIN_DET_CONF, MIN_TRK_CONF = 0.5, 0.5

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ========= 유틸: MediaPipe 21 → DHG/SHREC 22 매핑 =========
# MediaPipe index 참고:
# 0:wrist, (thumb)1:CMC,2:MCP,3:IP,4:tip, (index)5:MP,6:PIP,7:DIP,8:tip,
# (middle)9:MP,10:PIP,11:DIP,12:tip, (ring)13:MP,14:PIP,15:DIP,16:tip,
# (pinky)17:MP,18:PIP,19:DIP,20:tip
#
# DHG/SHREC 22 관절은 "wrist(0), palm center(1), 각 손가락 4개 관절" 컨벤션이 흔함.
# 아래는 근사 매핑(피처 추출 목적). 정확한 지도학습 재현이 목적이면 데이터셋 정식 매핑을 사용하세요.
def mediapipe21_to_dhg22(xyz21):
    """
    xyz21: (21, 3) numpy, [wrist, 20 joints], 단위는 픽셀/정규화 좌표 아무거나 상관없음(동일 스케일)
    return: (22, 3) numpy, [wrist, palm_center, 5 fingers x 4 joints]
    """
    wrist = xyz21[0]

    # MCP(각 손가락의 관절 시작점들): index,middle,ring,pinky 의 MCP + thumb의 MCP
    mcp_idx = [2, 5, 9, 13, 17]  # thumb MCP=2, 나머지 MCP들
    palm_center = (xyz21[[0] + mcp_idx].mean(axis=0))  # wrist + 5 MCP 평균으로 근사

    # 손가락별 4관절 구성: [MCP, PIP(or IP), DIP(or tip-1), tip]
    thumb = np.stack([xyz21[2], xyz21[3], xyz21[4], xyz21[1]], axis=0)   # MCP, IP, tip, CMC(근사)
    indexf = np.stack([xyz21[5], xyz21[6], xyz21[7], xyz21[8]], axis=0)
    middle = np.stack([xyz21[9], xyz21[10], xyz21[11], xyz21[12]], axis=0)
    ring   = np.stack([xyz21[13], xyz21[14], xyz21[15], xyz21[16]], axis=0)
    pinky  = np.stack([xyz21[17], xyz21[18], xyz21[19], xyz21[20]], axis=0)

    out = np.concatenate([
        wrist[None, :],
        palm_center[None, :],
        thumb, indexf, middle, ring, pinky
    ], axis=0)  # (22,3)
    return out

def normalize_xyz(xyz, eps=1e-6):
    """
    간단 정규화: (1) 기준점(손목) 원점 이동, (2) 손 크기 스케일 정규화.
    - 스케일: wrist->middle_mcp 거리 사용.
    """
    xyz = xyz.copy()
    wrist = xyz[0]
    xyz -= wrist
    # 스케일: 중지 MCP(9)까지의 거리(혹은 palm_center까지) 사용
    ref = xyz[2]  # thumb MCP를 쓰면 편향될 수 있어 middle MCP를 권장하지만, 위 매핑으로는 index가 달라짐
    # 매핑 후 22 포맷: wrist=0, palm=1, thumb MCP=2, index MCP=6, middle MCP=10, ...
    # 중지 MCP는 10
    ref = xyz[10]
    scale = np.linalg.norm(ref) + eps
    xyz /= scale
    return xyz

# ========= TD-GCN 로더 =========
def build_tdgcn_and_load(weights_path, config_yaml, device):
    sys.path.append(TDGCN_REPO)
    # config 로드(입력 채널/관절 수/클래스 수 등)
    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    # 모델 클래스 경로는 저장소 구조를 따름(대개 model/tdgcn.py 안 Model)
    # import 경로는 저장소에 따라 다를 수 있어 try 순회
    model = None
    tried = [
        ("model.tdgcn", "Model"),
        ("model.model", "Model"),
        ("model.tdgcn", "TDGCN"),
    ]
    last_err = None
    for mod_name, cls_name in tried:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            ModelClass = getattr(mod, cls_name)
            model = ModelClass(**cfg.get("model_args", {}))
            break
        except Exception as e:
            last_err = e
            continue
    if model is None:
        raise RuntimeError(f"TD-GCN 모델 임포트 실패: {last_err}")

    ckpt = torch.load(weights_path, map_location="cpu")
    # 일반적으로 'model_state_dict' 또는 직접 state_dict 저장
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    # 분류기 직전 피처 뽑기 위해 forward hook
    feature_blob = {"feat": None}
    # 보통 최종 fc 직전의 글로벌풀 출력 텐서 이름이 model.fc 입력임.
    # 구현체에 따라 모듈명이 다르므로 가장 마지막 nn.Linear 를 찾아 hook
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        # Linear가 없다면 글로벌풀 직전 텐서가 있는 모듈에 별도 훅이 필요
        # 여기서는 예외로 처리
        print("[WARN] 마지막 Linear 레이어를 찾지 못했어요. logits을 피처로 사용합니다.")
        return model, feature_blob, None

    def _hook(module, inputs):
        # inputs는 튜플 -> 분류기 입력 텐서 하나가 들어옵니다.
        x = inputs[0]
        # 배치 차원(보통 1) 제거해서 (feature_dim,)으로 저장
        # 필요에 따라 squeeze는 조정 가능
        feature_blob["feat"] = x.detach().clone().cpu()

    last_linear.register_forward_pre_hook(_hook)
    return model, feature_blob, last_linear

def to_tdgcn_input(xyz_seq_22):
    """
    xyz_seq_22: numpy (T, 22, 3)
    TD-GCN/CTR-GCN류 표준 입력: (N, C, T, V, M)
      - N: batch=1
      - C: 채널=3(x,y,z)
      - T: 프레임수
      - V: 관절수=22
      - M: 인스턴스 수(사람 수)=1
    """
    x = torch.from_numpy(xyz_seq_22.astype(np.float32))  # (T,22,3)
    x = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)    # (1,3,T,22,1)
    return x

def pick_hand(results):
    # 오른손 우선, 없으면 왼손
    if not results.multi_hand_landmarks:
        return None, None
    label = None
    idx = 0
    if results.multi_handedness:
        # 'Right' 우선
        for i, h in enumerate(results.multi_handedness):
            if h.classification[0].label == 'Right':
                label, idx = 'Right', i
                break
        else:
            label, idx = results.multi_handedness[0].classification[0].label, 0
    return label, results.multi_hand_landmarks[idx]

def landmarks_to_world_xyz21(hand_landmarks, w, h):
    # 여기서는 픽셀 좌표(x,y) + z(미디어파이프 normalized)를 사용
    pts = []
    for lm in hand_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        z = lm.z * max(w, h)  # 스케일을 대략 맞추기 위해 화면 크기 곱
        pts.append((x, y, z))
    return np.array(pts, dtype=np.float32)  # (21,3)

def main():
    device = torch.device("cuda" if USE_GPU else "cpu")
    model, feat_blob, last_linear = build_tdgcn_and_load(WEIGHTS_PATH, CONFIG_YAML, device)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2, model_complexity=1,
        min_detection_confidence=MIN_DET_CONF, min_tracking_confidence=MIN_TRK_CONF
    )

    seq_buf = deque(maxlen=SEQ_LEN)  # 각 원소는 (22,3) numpy
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

            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        out, hl, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # 한 손 선택 후 버퍼에 넣기
            label, hl = pick_hand(results)
            if hl is not None:
                xyz21 = landmarks_to_world_xyz21(hl, w, h)        # (21,3)
                xyz22 = mediapipe21_to_dhg22(xyz21)               # (22,3)
                xyz22 = normalize_xyz(xyz22)                      # 정규화
                seq_buf.append(xyz22)

            # 1초마다, 그리고 버퍼가 가득 찼을 때 피처 추출
            now = time.time()
            if now - last_print >= PRINT_INTERVAL and len(seq_buf) == SEQ_LEN:
                last_print = now
                seq_np = np.stack(list(seq_buf), axis=0)          # (T,22,3)
                x = to_tdgcn_input(seq_np).to(device)             # (1,3,T,22,1)

                with torch.no_grad():
                    logits = model(x)                              # (1, num_classes)
                    if feat_blob["feat"] is not None:
                        feat = feat_blob["feat"].cpu().numpy().squeeze(0)  # (feature_dim)
                    else:
                        # 훅 실패 시 logits을 임시 피처로 사용
                        feat = logits.cpu().numpy().squeeze(0)

                # 콘솔 출력(요약)
                print("=== TD-GCN Feature @ {:.1f}s ===".format(now))
                print(" seq_len:", len(seq_buf), "| hand:", label if label else "None")
                print(" feature_dim:", feat.shape[0])
                # 앞쪽 일부만 미리보기
                print(" feature[:10]:", np.array2string(feat[:10], precision=4, separator=", "))

            cv2.putText(out, f"T={len(seq_buf)}/{SEQ_LEN}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("Hands + TD-GCN Features (press 'q' to quit)", out)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()