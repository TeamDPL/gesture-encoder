#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2O Dataset 기반 Gesture Encoder Evaluation 스크립트
- kNN Accuracy
- Silhouette Score
- Smoothness(TS)
- Joint-Drop ΔAcc
결과는 모두 하나의 텍스트 파일(result/eval_summary.txt)에 정리해서 저장하고,
임베딩/라벨은 .npy로 저장함.
"""

import os
import glob
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier

# ===== TD-GCN + AUX(wrist) Encoder (H2O 평가용) =====

TDGCN_REPO   = os.path.expanduser("./TD-GCN-Gesture")
CONFIG_YAML  = os.path.join(TDGCN_REPO, "config", "dhg14-28", "DHG14-28.yaml")
WEIGHTS_PATH = os.path.join(TDGCN_REPO, "checkpoints", "DHG", "DHG14label", "Sub3_j.pt")

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
        feature_blob["feat"] = inputs[0].detach()  # (B,D)
    last_linear.register_forward_pre_hook(_hook)

    return model, feature_blob, last_linear


def build_wrist_aux(seq_wrist: np.ndarray) -> np.ndarray:
    """
    seq_wrist: (T,3)  →  aux: [mean(3), std(3)] = (6,)
    H2O에서는 world/px 개념이 없으니 그냥 joint0 trajectory 기준으로 사용.
    """
    if seq_wrist.shape[0] == 0:
        return np.zeros((6,), dtype=np.float32)
    mu = seq_wrist.mean(axis=0)
    sd = seq_wrist.std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)


class TDGCN_WristAux_Encoder(nn.Module):
    """
    H2O 평가용 TD-GCN 인코더 + AUX(wrist)
    입력: x (B,3,T,22,1)
    출력: z_cat (B, D+6)
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, self.feature_blob, _ = build_tdgcn_and_load(
            WEIGHTS_PATH, CONFIG_YAML, device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,T,22,1)
        """
        B, C, T, J, _ = x.shape

        # 1) encoder 통과
        with torch.no_grad():
            logits = self.model(x)  # (B,num_classes)
            if self.feature_blob["feat"] is not None:
                enc = self.feature_blob["feat"]  # (B,D)
            else:
                enc = logits                     # fallback

        # 2) AUX(wrist): 입력 x에서 joint0 trajectory를 뽑아서 mean/std 계산
        #    x: (B,3,T,22,1) → (B,T,22,3)
        x_np = (
            x.permute(0, 2, 3, 1, 4)  # (B,T,J,C,1)
             .contiguous()
             .view(B, T, J, C)        # (B,T,22,3)
             .cpu()
             .numpy()
        )

        aux_list = []
        for b in range(B):
            wrist_seq = x_np[b, :, 0, :]      # (T,3) 0번 joint = wrist
            aux = build_wrist_aux(wrist_seq)  # (6,)
            aux_list.append(aux)

        aux_arr = np.stack(aux_list, axis=0)          # (B,6)
        aux = torch.from_numpy(aux_arr).to(enc.device)  # (B,6)

        z_cat = torch.cat([enc, aux], dim=1)  # (B, D+6)
        return z_cat

# =====================
# 1. H2O pose 데이터셋 로더
# =====================

TRAIN_SEQS = [
    'subject1/h1', 'subject1/h2', 'subject1/k1', 'subject1/k2', 'subject1/o1', 'subject1/o2',
    'subject2/h1', 'subject2/h2', 'subject2/k1', 'subject2/k2', 'subject2/o1', 'subject2/o2',
    'subject3/h1', 'subject3/h2', 'subject3/k1'
]
VAL_SEQS = ['subject3/k2', 'subject3/o1', 'subject3/o2']
TEST_SEQS = ['subject4/h1', 'subject4/h2', 'subject4/k1', 'subject4/k2', 'subject4/o1', 'subject4/o2']


def load_sequence_poses(seq_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    한 시퀀스(예: subject1/h1/0/cam4)에 대해 hand pose + verb/action label을 읽어서 반환.

    NOTE:
    - 실제 H2O hand_pose/verb_label 파일 포맷에 맞게 이 부분은 조정해야 함.
    - 여기서는 hand_pose/*.txt, verb_label/*.txt 구조를 가정한 예시 코드.

    Returns:
        poses: (F, 2, 21, 3) float32, [left/right, 21 joints, xyz]
        labels: (F,) int64
    """
    hand_pose_dir = os.path.join(seq_root, "hand_pose")
    verb_label_dir = os.path.join(seq_root, "verb_label")

    pose_files = sorted(glob.glob(os.path.join(hand_pose_dir, "*.txt")))
    assert len(pose_files) > 0, f"No hand_pose txt found in {hand_pose_dir}"

    poses = []
    labels = []

    for pf in pose_files:
        arr = np.loadtxt(pf, dtype=np.float32)  # (1 + 21*3 + 1 + 21*3,) 가정

        # left
        left_valid = int(arr[0])  # 필요시 사용
        left_xyz = arr[1:1 + 21 * 3].reshape(21, 3)

        # right
        right_valid = int(arr[1 + 21 * 3])
        right_xyz = arr[1 + 21 * 3 + 1:].reshape(21, 3)

        pose_lr = np.stack([left_xyz, right_xyz], axis=0)  # (2,21,3)
        poses.append(pose_lr)

        fname = os.path.basename(pf)
        verb_file = os.path.join(verb_label_dir, fname)
        verb = int(np.loadtxt(verb_file, dtype=np.int64))
        labels.append(verb)

    poses = np.stack(poses, axis=0)         # (F,2,21,3)
    labels = np.array(labels, dtype=np.int64)  # (F,)
    return poses, labels


def h2o_sequences_for_split(root: str, split: str) -> List[str]:
    """
    split(train/val/test)에 해당하는 cam4 시퀀스 경로 리스트 반환.
    예: <root>/subject1/h1/0/cam4
    """
    if split == "train":
        base_list = TRAIN_SEQS
    elif split == "val":
        base_list = VAL_SEQS
    elif split == "test":
        base_list = TEST_SEQS
    else:
        raise ValueError(f"Unknown split: {split}")

    seq_dirs = []
    for subj_seq in base_list:
        seq_root = os.path.join(root, subj_seq)  # e.g. <root>/subject1/h1
        inner = sorted(
            d for d in glob.glob(os.path.join(seq_root, "*"))
            if os.path.isdir(d)
        )
        for idx_dir in inner:
            cam4 = os.path.join(idx_dir, "cam4")
            if os.path.isdir(cam4):
                seq_dirs.append(cam4)

    return seq_dirs


# =====================
# 2. 시퀀스 → 윈도우 단위 Dataset
# =====================

class H2OEncoderEvalDataset(Dataset):
    """
    H2O 시퀀스를 sliding window로 잘라 encoder 입력 형태로 만드는 Dataset.

    - poses: (F, 2, 21, 3)  → 오른손(또는 왼손)만 떼어 (T,21,3)
    - labels: (F,) → window 구간의 center frame label 또는 majority label 사용.
    """

    def __init__(
        self,
        root: str,
        split: str,
        seq_len: int,
        stride: int,
        use_right_hand: bool,
        label_mode: str,
        preprocess_fn,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.use_right_hand = use_right_hand
        self.label_mode = label_mode
        self.preprocess_fn = preprocess_fn

        self.samples: List[Dict] = []

        seq_dirs = h2o_sequences_for_split(root, split)
        print(f"[H2O] {split}: {len(seq_dirs)} sequences")

        for cam4_dir in tqdm(seq_dirs, desc=f"Load {split}"):
            poses, labels = load_sequence_poses(cam4_dir)  # (F,2,21,3), (F,)

            hand_idx = 1 if use_right_hand else 0
            hand_xyz = poses[:, hand_idx]  # (F,21,3)

            F = hand_xyz.shape[0]
            for start in range(0, max(F - seq_len + 1, 1), stride):
                end = start + seq_len
                if end > F:
                    continue

                xyz = hand_xyz[start:end]      # (T,21,3)
                seq_label = labels[start:end]  # (T,)

                if label_mode == "center":
                    label = int(seq_label[len(seq_label) // 2])
                else:  # majority
                    vals, cnt = np.unique(seq_label, return_counts=True)
                    label = int(vals[np.argmax(cnt)])

                self.samples.append(
                    dict(
                        xyz=xyz.astype(np.float32),
                        label=label,
                    )
                )

        print(f"[H2O] {split}: {len(self.samples)} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xyz = self.samples[idx]["xyz"]  # (T,21,3)
        x = self.preprocess_fn(xyz)     # torch.Tensor, (C,T,J,1)
        y = self.samples[idx]["label"]
        return x, y


# =====================
# 3. Encoder 평가 유틸 함수들
# =====================

@torch.no_grad()
def compute_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    DataLoader 전체에 대해 encoder 임베딩과 라벨을 계산.

    Returns:
        embeddings: (N, D)
        labels: (N,)
    """
    encoder.eval()
    all_emb = []
    all_lab = []

    for x, y in tqdm(loader, desc="Encoding"):
        x = x.to(device, non_blocking=True)  # (B,C,T,J,1)
        z = encoder(x)                       # (B,D) 라고 가정
        all_emb.append(z.cpu().numpy())
        all_lab.append(y.numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    labels = np.concatenate(all_lab, axis=0)
    return embeddings, labels


def evaluate_knn_accuracy(
    train_emb: np.ndarray,
    train_lab: np.ndarray,
    test_emb: np.ndarray,
    test_lab: np.ndarray,
    k: int = 5,
) -> float:
    """
    k-NN classifier 로 encoder 분별력 평가.
    """
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(train_emb, train_lab)
    pred = knn.predict(test_emb)
    acc = (pred == test_lab).mean()
    return float(acc)


def evaluate_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette Score: 클러스터 분리도. 값이 클수록 class 간 분리 잘 됨.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(embeddings, labels, metric="euclidean"))


def evaluate_smoothness_ts(embeddings: np.ndarray, seq_ids: np.ndarray) -> float:
    """
    Smoothness(TS):
    - 동일 "sequence id" 내에서 연속 임베딩의 cosine similarity 평균.

    여기서는 간단히 seq_ids 를 이용해 그룹핑하는 형태.
    실제로는 frame-level/sequence-level 설계에 따라 조정 가능.
    """
    from sklearn.preprocessing import normalize

    z = normalize(embeddings)  # (N,D) L2 normalize
    ts_vals = []

    unique_seq = np.unique(seq_ids)
    for sid in unique_seq:
        idx = np.where(seq_ids == sid)[0]
        if len(idx) < 2:
            continue
        seq_z = z[idx]  # (L,D)
        sim = (seq_z[:-1] * seq_z[1:]).sum(axis=1)  # cos sim
        ts_vals.append(sim.mean())

    if not ts_vals:
        return float("nan")
    return float(np.mean(ts_vals))


def apply_joint_drop(x: torch.Tensor, drop_prob: float = 0.3) -> torch.Tensor:
    """
    Joint-Drop augmentation:
    - 각 샘플마다 일정 확률로 joint 를 0으로 만들어 occlusion 시뮬레이션.

    Args:
        x: (B,C,T,J,1)
    Returns:
        x_occluded: (B,C,T,J,1)
    """
    B, C, T, J, _ = x.shape
    # joint 단위 mask: (B,1,1,J,1) — 전체 시간에 대해 같은 joint 를 drop
    mask = (torch.rand(B, 1, 1, J, 1, device=x.device) > drop_prob).float()
    return x * mask


@torch.no_grad()
def evaluate_joint_drop_delta_acc(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    k: int = 5,
    drop_prob: float = 0.3,
) -> float:
    """
    Joint-Drop Robustness (ΔAcc):
    - clean 버전 k-NN accuracy
    - joint drop 적용 후 다시 k-NN accuracy
    - ΔAcc = clean_acc - occluded_acc
    """

    # 1) clean embeddings
    print("[ΔAcc] Computing clean embeddings...")
    train_emb, train_lab = compute_embeddings(encoder, train_loader, device)
    test_emb, test_lab = compute_embeddings(encoder, test_loader, device)
    clean_acc = evaluate_knn_accuracy(train_emb, train_lab, test_emb, test_lab, k=k)
    print(f"[ΔAcc] Clean k-NN@{k} Acc = {clean_acc:.4f}")

    # 2) occluded embeddings (joint drop)
    print(f"[ΔAcc] Computing occluded embeddings (drop_prob={drop_prob})...")
    encoder.eval()
    all_train_emb, all_train_lab = [], []
    for x, y in tqdm(train_loader, desc="Encoding train (joint-drop)"):
        x = x.to(device, non_blocking=True)
        x_occ = apply_joint_drop(x, drop_prob=drop_prob)
        z = encoder(x_occ)
        all_train_emb.append(z.cpu().numpy())
        all_train_lab.append(y.numpy())
    train_occ_emb = np.concatenate(all_train_emb, axis=0)
    train_occ_lab = np.concatenate(all_train_lab, axis=0)

    all_test_emb, all_test_lab = [], []
    for x, y in tqdm(test_loader, desc="Encoding test (joint-drop)"):
        x = x.to(device, non_blocking=True)
        x_occ = apply_joint_drop(x, drop_prob=drop_prob)
        z = encoder(x_occ)
        all_test_emb.append(z.cpu().numpy())
        all_test_lab.append(y.numpy())
    test_occ_emb = np.concatenate(all_test_emb, axis=0)
    test_occ_lab = np.concatenate(all_test_lab, axis=0)

    occ_acc = evaluate_knn_accuracy(train_occ_emb, train_occ_lab, test_occ_emb, test_occ_lab, k=k)
    print(f"[ΔAcc] Occluded k-NN@{k} Acc = {occ_acc:.4f}")

    delta_acc = clean_acc - occ_acc
    return float(delta_acc)


# =====================
# 4. Main: 전체 평가 파이프라인
# =====================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--h2o_root", type=str, required=True, help="H2O pose root directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--drop_prob", type=float, default=0.3)
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ===== 4-1. encoder 로드 (여기 부분을 실제 TD-GCN encoder 로 교체하면 됨) =====
    class DummyEncoder(nn.Module):
        def __init__(self, in_dim=3 * 64 * 21, dim=256):
            super().__init__()
            self.fc = nn.Linear(in_dim, dim)
        def forward(self, x):
            B, C, T, J, _ = x.shape
            x = x.view(B, C * T * J)
            return self.fc(x)

    encoder = TDGCN_WristAux_Encoder(device)
    print("[Encoder] TD-GCN + AUX(wrist) encoder loaded")

    # ===== 4-2. preprocess 함수 =====
    def default_preprocess(xyz: np.ndarray) -> torch.Tensor:
        """
        xyz: (T,J,3)  -- H2O pose 시퀀스 (오른손만 있다고 가정)
        전제:
        - J가 이미 TD-GCN에서 쓰는 22개 joint 순서(DHG22)와 동일하다고 가정.
        - 만약 21개라면, mediapipe21_to_dhg22처럼 palm_center 추가해서 22로 맞춰야 함.

        여기서는:
        1) 시퀀스 전체를 하나의 origin(첫 프레임 wrist) 기준으로 shift
        2) 첫 프레임 wrist→middle MCP 거리로 scale normalize
        3) (T,22,3) → (3,T,22,1)
        """
        xyz22 = xyz.astype(np.float32)  # (T,J,3), J=22 가정

        # origin: 첫 프레임 wrist(0번 joint)
        origin = xyz22[0, 0].copy()              # (3,)
        xyz22 -= origin                          # (T,J,3)

        # scale: 첫 프레임에서 wrist(0)→middle MCP(10) 거리
        scale = np.linalg.norm(xyz22[0, 10]) + 1e-6
        xyz22 /= scale

        # TD-GCN 입력 포맷으로 변환
        x = torch.from_numpy(xyz22).permute(2, 0, 1).unsqueeze(-1)  # (3,T,22,1)
        return x

    # ===== 4-3. Dataset & DataLoader =====
    train_set = H2OEncoderEvalDataset(
        root=args.h2o_root,
        split="train",
        seq_len=args.seq_len,
        stride=args.stride,
        use_right_hand=True,
        label_mode="center",
        preprocess_fn=default_preprocess,
    )
    test_set = H2OEncoderEvalDataset(
        root=args.h2o_root,
        split="test",
        seq_len=args.seq_len,
        stride=args.stride,
        use_right_hand=True,
        label_mode="center",
        preprocess_fn=default_preprocess,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ===== 4-4. result 폴더 & summary 파일 준비 =====
    os.makedirs("result", exist_ok=True)
    summary_path = os.path.join("result", "eval_summary.txt")
    summary_lines = []

    # ===== 4-5. 임베딩 계산 및 저장 =====
    print("\n[Eval] Computing embeddings (train/test) ...")
    train_emb, train_lab = compute_embeddings(encoder, train_loader, device)
    test_emb, test_lab = compute_embeddings(encoder, test_loader, device)

    np.save(os.path.join("result", "embeddings_train.npy"), train_emb)
    np.save(os.path.join("result", "embeddings_test.npy"), test_emb)
    np.save(os.path.join("result", "labels_train.npy"), train_lab)
    np.save(os.path.join("result", "labels_test.npy"), test_lab)

    summary_lines.append(f"Embeddings Train Shape: {train_emb.shape}")
    summary_lines.append(f"Embeddings Test Shape: {test_emb.shape}")
    summary_lines.append(f"Labels Train Shape: {train_lab.shape}")
    summary_lines.append(f"Labels Test Shape: {test_lab.shape}")

    # ===== 4-6. k-NN Accuracy =====
    knn_acc = evaluate_knn_accuracy(train_emb, train_lab, test_emb, test_lab, k=args.k)
    print(f"[Eval] k-NN@{args.k} Accuracy = {knn_acc:.6f}")
    summary_lines.append(f"k-NN Accuracy (k={args.k}): {knn_acc:.6f}")

    # ===== 4-7. Silhouette =====
    all_emb = np.concatenate([train_emb, test_emb], axis=0)
    all_lab = np.concatenate([train_lab, test_lab], axis=0)
    sil = evaluate_silhouette(all_emb, all_lab)
    print(f"[Eval] Silhouette Score = {sil:.6f}")
    summary_lines.append(f"Silhouette Score: {sil:.6f}")

    # ===== 4-8. Smoothness(TS) =====
    # 여기서는 간단하게 train 내 샘플 index를 seq_id 로 사용
    seq_ids = np.arange(len(train_emb))
    ts = evaluate_smoothness_ts(train_emb, seq_ids)
    print(f"[Eval] Smoothness (TS) = {ts:.6f}")
    summary_lines.append(f"Smoothness (TS): {ts:.6f}")

    # ===== 4-9. Joint-Drop ΔAcc =====
    delta_acc = evaluate_joint_drop_delta_acc(
        encoder, train_loader, test_loader, device,
        k=args.k, drop_prob=args.drop_prob
    )
    print(f"[Eval] Joint-Drop ΔAcc = {delta_acc:.6f}")
    summary_lines.append(f"Joint-Drop ΔAcc (drop_prob={args.drop_prob}): {delta_acc:.6f}")

    # ===== 4-10. Summary 파일로 저장 =====
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"\n[Saved] Evaluation summary → {summary_path}")


if __name__ == "__main__":
    main()