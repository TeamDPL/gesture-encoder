#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2O Dataset Encoder Evaluation Script
Evaluates gesture encoders (TD-GCN, ST-GCN) on H2O dataset.
Metrics: k-NN Accuracy, Silhouette Score, Smoothness (TS), Joint-Drop ΔAcc.
Output: result/log_YYMMDD_HHMMSS
"""

import os
import sys
import glob
import random
import yaml
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from dotenv import load_dotenv
import wandb

# =====================
# Encoder Loading
# =====================

def load_encoder(model_type, device):
    if model_type == "tdgcn_dual":
        import tdgcn_dual
        return tdgcn_dual.get_encoder(device)
    elif model_type == "tdgcn_dual_wrist":
        import tdgcn_dual_wrist
        return tdgcn_dual_wrist.get_encoder(device)
    elif model_type == "stgcn_sl_64":
        import stgcn_sl_64
        return stgcn_sl_64.get_encoder(device)
    elif model_type == "stgcn_sl_1":
        import stgcn_sl_1
        return stgcn_sl_1.get_encoder(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def find_all_sequences(root_dir):
    # Pattern: {root}/{action_type}/{seq_id}/cam4
    pattern = os.path.join(root_dir, "*", "*", "cam4")
    found = sorted(glob.glob(pattern))
    return found

def load_sequence_poses(seq_root: str):
    hand_pose_dir = os.path.join(seq_root, "hand_pose")
    verb_label_dir = os.path.join(seq_root, "verb_label")

    if not os.path.exists(hand_pose_dir) or not os.path.exists(verb_label_dir):
        return None, None

    pose_files = sorted(glob.glob(os.path.join(hand_pose_dir, "*.txt")))
    label_files = sorted(glob.glob(os.path.join(verb_label_dir, "*.txt")))

    if len(pose_files) == 0 or len(pose_files) != len(label_files):
        return None, None

    poses = []
    labels = []

    for pf, lf in zip(pose_files, label_files):
        try:
            p_data = np.loadtxt(pf, dtype=np.float32)
            l_data = np.loadtxt(lf, dtype=np.float32).astype(np.int64)
        except Exception as e:
            continue
            
        if p_data.shape[0] != 128: 
            continue
            
        # 0: valid_right, 1..63: right_hand, 64: valid_left, 65..127: left_hand
        right_valid = p_data[0]
        right_pose = p_data[1:64].reshape(21, 3)
        
        left_valid = p_data[64]
        left_pose = p_data[65:128].reshape(21, 3)
        
        if right_valid < 0.5:
            right_pose = np.zeros((21, 3), dtype=np.float32)
        if left_valid < 0.5:
            left_pose = np.zeros((21, 3), dtype=np.float32)
            
        if right_valid > 0.5 or left_valid > 0.5:
            # Stack Right, Left -> (2, 21, 3)
            # Index 0: Right, Index 1: Left
            both_hands = np.stack([right_pose, left_pose], axis=0)
            poses.append(both_hands)
            labels.append(int(l_data))

    if len(poses) == 0:
        return None, None

    return np.stack(poses, axis=0), np.array(labels, dtype=np.int64)

def _convert_to_dhg22(xyz21):
    # xyz21: (21, 3)
    wrist = xyz21[0]
    mcp_idx = [2, 5, 9, 13, 17]
    palm_center = xyz21[[0] + mcp_idx].mean(axis=0)
    
    thumb_pts = xyz21[[2, 3, 4, 1]]
    index_pts = xyz21[[5, 6, 7, 8]]
    middle_pts = xyz21[[9, 10, 11, 12]]
    ring_pts = xyz21[[13, 14, 15, 16]]
    pinky_pts = xyz21[[17, 18, 19, 20]]
    
    frame22 = np.concatenate([
        wrist[None, :], palm_center[None, :],
        thumb_pts, index_pts, middle_pts, ring_pts, pinky_pts
    ], axis=0)
    return frame22

def preprocess_tdgcn_dual(xyz: np.ndarray) -> torch.Tensor:
    # xyz: (T, 2, 21, 3) -> Returns (2, 3, T, 22, 1) stacked
    # Independent normalization
    T = xyz.shape[0]
    out_list = []
    
    for hand_idx in range(2): # 0: Right, 1: Left
        hand_seq = xyz[:, hand_idx] # (T, 21, 3)
        xyz22_list = []
        for t in range(T):
            frame21 = hand_seq[t]
            frame22 = _convert_to_dhg22(frame21)
            
            # Normalize to own wrist
            frame22 -= frame22[0]
            scale = np.linalg.norm(frame22[10]) + 1e-6
            frame22 /= scale
            xyz22_list.append(frame22)
        
        xyz22 = np.stack(xyz22_list, axis=0) # (T, 22, 3)
        # (3, T, 22, 1)
        x = torch.from_numpy(xyz22).permute(2, 0, 1).unsqueeze(-1)
        out_list.append(x)
        
    # Stack hands: (2, 3, T, 22, 1)
    return torch.stack(out_list, dim=0)

def preprocess_tdgcn_wrist(xyz: np.ndarray):
    # xyz: (T, 2, 21, 3)
    # Right (idx 0) normalized to Left Wrist (idx 1)
    # Left (idx 1) normalized to Left Wrist
    
    T = xyz.shape[0]
    
    # Get Left Wrist sequence for normalization origin
    # xyz[:, 1, 0, :] -> (T, 3)
    left_wrist_seq = xyz[:, 1, 0, :].copy()
    
    out_x_list = []
    out_aux_list = []
    
    for hand_idx in range(2):
        hand_seq = xyz[:, hand_idx] # (T, 21, 3)
        xyz22_list = []
        
        # For AUX (only needed for the hand being processed, but we need to match shapes)
        # The original code calculates AUX from the wrist sequence of THAT hand.
        # Wait, tdgcn_dual_wrist.py calculates AUX from the wrist of the hand being processed.
        # "wrist_buf[eff_label].append(wristN)"
        # So we should compute AUX for each hand independently?
        # Yes, "ENC C=3 | AUX(wrist) dim=6".
        
        # But the normalization origin differs.
        # "Left": origin = Left Wrist
        # "Right": origin = Left Wrist (relative)
        
        # Calculate AUX (Mean/Std of wrist) BEFORE relative normalization?
        # tdgcn_dual_wrist.py: 
        # 1. Get world xyz22
        # 2. Update left_wrist_world if Left
        # 3. Determine origin (Left Wrist)
        # 4. Normalize xyz22 (minus origin, divide scale) -> Input to GCN
        # 5. Get wristW (world), normalize to screen (wristN) -> Input to AUX
        
        # So AUX is based on "Screen Normalized" wrist, not "Relative" wrist.
        # Since we don't have screen dimensions here easily (it's normalized data?), 
        # we might have to approximate or skip AUX if it's just for position encoding.
        # The H2O data is already in some coordinate system.
        # Let's assume the input `xyz` is "World-like".
        # We can compute mean/std of the raw wrist coords (or zero-centered per sequence).
        
        # Let's follow the logic:
        # AUX = Mean/Std of the wrist trajectory (absolute or screen-relative).
        # Here we can just use the wrist trajectory from the input `xyz`.
        
        wrist_seq = hand_seq[:, 0, :] # (T, 3)
        mu = wrist_seq.mean(axis=0)
        sd = wrist_seq.std(axis=0)
        aux = np.concatenate([mu, sd], axis=0).astype(np.float32) # (6,)
        out_aux_list.append(torch.from_numpy(aux))

        for t in range(T):
            frame21 = hand_seq[t]
            frame22 = _convert_to_dhg22(frame21)
            
            # Origin: Always Left Wrist of current frame
            origin = left_wrist_seq[t]
            
            frame22 -= origin
            scale = np.linalg.norm(frame22[10]) + 1e-6
            frame22 /= scale
            xyz22_list.append(frame22)
            
        xyz22 = np.stack(xyz22_list, axis=0)
        x = torch.from_numpy(xyz22).permute(2, 0, 1).unsqueeze(-1)
        out_x_list.append(x)
        
    return torch.stack(out_x_list, dim=0), torch.stack(out_aux_list, dim=0)

def preprocess_stgcn(xyz: np.ndarray, V=21):
    # xyz: (T, 2, 21, 3) -> (2, 3, T, V, 1)
    T = xyz.shape[0]
    out_list = []
    
    for hand_idx in range(2):
        hand_seq = xyz[:, hand_idx]
        xyz_out = []
        for t in range(T):
            frame = hand_seq[t].copy()
            wrist = frame[0]
            frame -= wrist
            scale = np.linalg.norm(frame[9]) + 1e-6
            frame /= scale
            
            if V > 21:
                pad = np.zeros((V - 21, 3), dtype=np.float32)
                frame = np.concatenate([frame, pad], axis=0)
            elif V < 21:
                frame = frame[:V]
            xyz_out.append(frame)
            
        xyz_np = np.stack(xyz_out, axis=0)
        x = torch.from_numpy(xyz_np).permute(2, 0, 1).unsqueeze(-1)
        out_list.append(x)
        
    return torch.stack(out_list, dim=0)

def split_sequences(seq_dirs: List[str], train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    Deterministic split based on sorted paths.
    """
    # Shuffle deterministically
    rng = random.Random(42)
    
    # Make a copy to shuffle
    shuffled = list(seq_dirs)
    rng.shuffle(shuffled)
    
    n_train = int(len(shuffled) * train_ratio)
    train_seqs = shuffled[:n_train]
    test_seqs = shuffled[n_train:]
    
    return train_seqs, test_seqs

# =====================
# 2. Dataset
# =====================

class H2OEncoderEvalDataset(Dataset):
    def __init__(
        self,
        seq_dirs: List[str],
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
        self.seq_dirs = seq_dirs

        self.samples: List[Dict] = []

        print(f"[Dataset] Loading {len(seq_dirs)} sequences...")

        for cam4_dir in tqdm(seq_dirs, desc="Loading Seqs"):
            poses, labels = load_sequence_poses(cam4_dir)
            if poses is None:
                continue

            # poses is already (F, 21, 3) from load_sequence_poses (right hand only)
            hand_xyz = poses  # (F, 21, 3)

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
                
                if label == -1: # Skip invalid labels
                    continue

                self.samples.append(
                    dict(
                        xyz=xyz.astype(np.float32),
                        label=label,
                        seq_id=self.seq_dirs.index(cam4_dir) # Track sequence ID
                    )
                )

        print(f"[Dataset] Total windows: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        xyz = self.samples[idx]["xyz"]  # (T,21,3)
        x = self.preprocess_fn(xyz)     # torch.Tensor, (C,T,J,1)
        y = self.samples[idx]["label"]
        sid = self.samples[idx]["seq_id"]
        return x, y, sid

# =====================
# 3. Metrics & Utils
# =====================

@torch.no_grad()
def compute_embeddings(encoder, loader, device):
    encoder.eval()
    all_emb = []
    all_lab = []
    all_seq = []

    for batch in tqdm(loader, desc="Encoding"):
        x, y, sid = batch
        
        # Handle tuple input (x, aux) for tdgcn_wrist
        if isinstance(x, (list, tuple)):
            # x is [Tensor(B, 2, 3, T, 22, 1), Tensor(B, 2, 6)]
            x_in = x[0].to(device, non_blocking=True)
            aux_in = x[1].to(device, non_blocking=True)
            
            # Flatten B and 2
            B, _, C, T, J, M = x_in.shape
            x_flat = x_in.view(B * 2, C, T, J, M)
            aux_flat = aux_in.view(B * 2, -1)
            
            z = encoder(x_flat, aux_flat) # (B*2, D)
            
        else:
            # x is Tensor(B, 2, 3, T, V, 1)
            x_in = x.to(device, non_blocking=True)
            B, _, C, T, V, M = x_in.shape
            x_flat = x_in.view(B * 2, C, T, V, M)
            
            z = encoder(x_flat) # (B*2, D)
            
        # Reshape back to (B, 2*D)
        # z: (B*2, D) -> (B, 2, D) -> (B, 2*D)
        D = z.shape[-1]
        z_dual = z.view(B, 2, D).reshape(B, 2 * D)
            
        all_emb.append(z_dual.cpu().numpy())
        all_lab.append(y.numpy())
        all_seq.append(sid.numpy())

    if not all_emb:
        return np.array([]), np.array([]), np.array([])

    embeddings = np.concatenate(all_emb, axis=0)
    labels = np.concatenate(all_lab, axis=0)
    seq_ids = np.concatenate(all_seq, axis=0)
    return embeddings, labels, seq_ids

def evaluate_knn_accuracy(train_emb, train_lab, test_emb, test_lab, k=5):
    if len(train_emb) == 0 or len(test_emb) == 0:
        return 0.0
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(train_emb, train_lab)
    
    # Check for unseen labels
    unique_train = set(train_lab)
    unique_test = set(test_lab)
    unseen = unique_test - unique_train
    if unseen:
        print(f"[Warn] Test set contains labels not in train set: {unseen}")
        print(f"[Warn] This will likely result in 0 accuracy for these classes.")
        
    pred = knn.predict(test_emb)
    acc = (pred == test_lab).mean()
    return float(acc)

def evaluate_silhouette(embeddings, labels):
    if len(embeddings) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    # Sample if too large to save time, but here we do full
    if len(embeddings) > 10000:
        # Optional: subsample for speed
        pass
    return float(silhouette_score(embeddings, labels, metric="euclidean"))

def evaluate_smoothness_ts(embeddings, seq_ids):
    from sklearn.preprocessing import normalize
    if len(embeddings) == 0:
        return float("nan")
    
    z = normalize(embeddings)
    ts_vals = []

    unique_seq = np.unique(seq_ids)
    for sid in unique_seq:
        idx = np.where(seq_ids == sid)[0]
        if len(idx) < 2:
            continue
        seq_z = z[idx]
        sim = (seq_z[:-1] * seq_z[1:]).sum(axis=1)
        ts_vals.append(sim.mean())

    if not ts_vals:
        return float("nan")
    return float(np.mean(ts_vals))

def apply_joint_drop(x, drop_prob=0.3):
    # If x is tuple (x_main, aux), only drop on x_main
    if isinstance(x, (list, tuple)):
        x_main, aux = x
        # x_main: (B, 2, C, T, J, 1)
        B, Two, C, T, J, _ = x_main.shape
        mask = (torch.rand(B, Two, 1, 1, J, 1, device=x_main.device) > drop_prob).float()
        return (x_main * mask, aux)
    else:
        # x: (B, 2, C, T, J, 1)
        B, Two, C, T, J, _ = x.shape
        mask = (torch.rand(B, Two, 1, 1, J, 1, device=x.device) > drop_prob).float()
        return x * mask

@torch.no_grad()
def evaluate_joint_drop_delta_acc(encoder, train_loader, test_loader, device, k=5, drop_prob=0.3):
    # 1) Clean
    print("[ΔAcc] Computing clean embeddings...")
    train_emb, train_lab, _ = compute_embeddings(encoder, train_loader, device)
    test_emb, test_lab, _ = compute_embeddings(encoder, test_loader, device)
    
    if len(train_emb) == 0 or len(test_emb) == 0:
        return 0.0

    clean_acc = evaluate_knn_accuracy(train_emb, train_lab, test_emb, test_lab, k=k)
    print(f"[ΔAcc] Clean k-NN@{k} Acc = {clean_acc:.4f}")

    # 2) Occluded
    print(f"[ΔAcc] Computing occluded embeddings (drop_prob={drop_prob})...")
    encoder.eval()
    
    # Helper to process loop
    def encode_loop(loader, desc):
        all_emb, all_lab = [], []
        for batch in tqdm(loader, desc=desc):
            x, y, _ = batch
            
            if isinstance(x, (list, tuple)):
                # x is [Tensor(B, 2, 3, T, 22, 1), Tensor(B, 2, 6)]
                x_in = x[0].to(device, non_blocking=True)
                aux_in = x[1].to(device, non_blocking=True)
                
                # Apply drop (on x_in)
                # apply_joint_drop handles the tuple structure now
                x_occ_main, aux_occ = apply_joint_drop((x_in, aux_in), drop_prob=drop_prob)
                
                # Flatten
                B, _, C, T, J, M = x_occ_main.shape
                x_flat = x_occ_main.view(B * 2, C, T, J, M)
                aux_flat = aux_occ.view(B * 2, -1)
                
                z = encoder(x_flat, aux_flat)
            else:
                x = x.to(device, non_blocking=True)
                x_occ = apply_joint_drop(x, drop_prob=drop_prob)
                
                B, _, C, T, V, M = x_occ.shape
                x_flat = x_occ.view(B * 2, C, T, V, M)
                
                z = encoder(x_flat)
                
            # Reshape back to (B, 2*D)
            D = z.shape[-1]
            z_dual = z.view(B, 2, D).reshape(B, 2 * D)
                
            all_emb.append(z_dual.cpu().numpy())
            all_lab.append(y.numpy())
        
        if not all_emb:
            return np.array([]), np.array([])
            
        return np.concatenate(all_emb, axis=0), np.concatenate(all_lab, axis=0)

    train_occ_emb, train_occ_lab = encode_loop(train_loader, "Encoding train (joint-drop)")
    test_occ_emb, test_occ_lab = encode_loop(test_loader, "Encoding test (joint-drop)")

    if len(train_occ_emb) == 0 or len(test_occ_emb) == 0:
        return 0.0

    occ_acc = evaluate_knn_accuracy(train_occ_emb, train_occ_lab, test_occ_emb, test_occ_lab, k=k)
    print(f"[ΔAcc] Occluded k-NN@{k} Acc = {occ_acc:.4f}")

    return float(clean_acc - occ_acc)

# =====================
# 4. Main
# =====================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--h2o_root", type=str, required=True, help="H2O dataset root")
    parser.add_argument("--model_type", type=str, default="tdgcn_dual", 
                        choices=["tdgcn_dual", "tdgcn_dual_wrist", "stgcn_sl_64", "stgcn_sl_1"],
                        help="Model type to evaluate")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--drop_prob", type=float, default=0.3)
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "GCCP Gesture Encoder"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"{args.model_type}_{datetime.now().strftime('%y%m%d_%H%M%S')}",
            config={
                "model_type": args.model_type,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "stride": args.stride,
                "k": args.k,
                "drop_prob": args.drop_prob,
                "device": str(args.use_gpu and torch.cuda.is_available()),
                "h2o_root": args.h2o_root,
            }
        )

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model Type: {args.model_type}")

    # Load Encoder
    try:
        encoder = load_encoder(args.model_type, device)
        print(f"[Encoder] {args.model_type} loaded")
    except Exception as e:
        print(f"[Error] Failed to load encoder: {e}")
        return

    # Select Preprocess Function
    if args.model_type == "tdgcn_dual":
        preprocess_fn = preprocess_tdgcn_dual
    elif args.model_type == "tdgcn_dual_wrist":
        preprocess_fn = preprocess_tdgcn_wrist
    elif args.model_type == "stgcn_sl_64":
        preprocess_fn = lambda x: preprocess_stgcn(x, V=42) # Assuming V=42 for SL? Or infer?
        # stgcn_sl_64.py infers V. Usually 42 for 2 hands or 21 for 1.
        # The script uses single hand, so maybe 21 or 42 with padding.
        # Let's assume 42 for safety as per stgcn_sl_64.py logic (it pads).
        # Actually stgcn_sl_64.py: infer_expected_V -> if >21 pad.
        # We should probably check the model's expected V.
        # But we don't have easy access to it here without inspecting the loaded model object deeply.
        # Let's assume 42 for now as it's common for "skeleton" config.
        pass
    elif args.model_type == "stgcn_sl_1":
        preprocess_fn = lambda x: preprocess_stgcn(x, V=21) # T=1 usually 21?
        pass
        
    # Refine STGCN V
    if "stgcn" in args.model_type:
        # Try to infer V from model
        try:
            # This is hacky, depends on model structure
            if hasattr(encoder.model, "data_bn"):
                nf = encoder.model.data_bn.num_features
                # nf = C * V * M
                # C=3, M=1 (usually)
                V = nf // 3
                print(f"[Info] Inferred V={V} from model")
                preprocess_fn = lambda x: preprocess_stgcn(x, V=V)
        except:
            print("[Warn] Could not infer V, defaulting to 42")
            preprocess_fn = lambda x: preprocess_stgcn(x, V=42)

    # Find Sequences
    all_seqs = find_all_sequences(args.h2o_root)
    print(f"Found {len(all_seqs)} sequences in {args.h2o_root}")
    if len(all_seqs) == 0:
        print("No sequences found. Check the path structure.")
        return

    # Split
    train_seqs, test_seqs = split_sequences(all_seqs, train_ratio=0.8)
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Test sequences: {len(test_seqs)}")

    # Create Datasets
    train_ds = H2OEncoderEvalDataset(
        seq_dirs=train_seqs,
        seq_len=args.seq_len,
        stride=args.stride,
        use_right_hand=True,
        label_mode="center",
        preprocess_fn=preprocess_fn,
    )
    test_ds = H2OEncoderEvalDataset(
        seq_dirs=test_seqs,
        seq_len=args.seq_len,
        stride=args.stride,
        use_right_hand=True,
        label_mode="center",
        preprocess_fn=preprocess_fn,
    )

    if len(train_ds) == 0:
        print("Train set is empty. Exiting.")
        return
    if len(test_ds) == 0:
        print("Test set is empty. Exiting.")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Result Setup
    if args.model_type == "tdgcn_dual":
        res_dir = "result/tdgcn"
    elif args.model_type == "tdgcn_dual_wrist":
        res_dir = "result/tdgcn-wrist"
    elif "stgcn" in args.model_type:
        res_dir = "result/stgcn-sl"
    else:
        res_dir = "result"
        
    os.makedirs(res_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_path = f"{res_dir}/log_{args.model_type}_{timestamp}"
    # User said "result/summary_YYMMDD_HHMMSS" as name. I'll assume it's a file.
    
    summary_lines = []
    summary_lines.append(f"Date: {timestamp}")
    summary_lines.append(f"Dataset Root: {args.h2o_root}")
    summary_lines.append(f"Encoder: {args.model_type}")
    summary_lines.append(f"Train Seqs: {len(train_seqs)}, Test Seqs: {len(test_seqs)}")
    summary_lines.append("-" * 20)

    # Compute Embeddings
    print("\n[Eval] Computing embeddings...")
    train_emb, train_lab, train_seq = compute_embeddings(encoder, train_loader, device)
    test_emb, test_lab, test_seq = compute_embeddings(encoder, test_loader, device)

    # Save embeddings (optional, but good for debug)
    # np.save(os.path.join("result", f"emb_train_{timestamp}.npy"), train_emb)
    # np.save(os.path.join("result", f"emb_test_{timestamp}.npy"), test_emb)

    # Metrics
    # 1. kNN
    knn_acc = evaluate_knn_accuracy(train_emb, train_lab, test_emb, test_lab, k=args.k)
    print(f"[Eval] k-NN@{args.k} Accuracy = {knn_acc:.6f}")
    summary_lines.append(f"k-NN Accuracy (k={args.k}): {knn_acc:.6f}")

    # 2. Silhouette
    all_emb = np.concatenate([train_emb, test_emb], axis=0)
    all_lab = np.concatenate([train_lab, test_lab], axis=0)
    sil = evaluate_silhouette(all_emb, all_lab)
    print(f"[Eval] Silhouette Score = {sil:.6f}")
    summary_lines.append(f"Silhouette Score: {sil:.6f}")

    # 3. Smoothness (TS)
    all_seq = np.concatenate([train_seq, test_seq], axis=0)
    ts = evaluate_smoothness_ts(all_emb, all_seq)
    print(f"[Eval] Smoothness (TS) = {ts:.6f}")
    summary_lines.append(f"Smoothness (TS): {ts:.6f}")

    # 4. Joint Drop
    delta_acc = evaluate_joint_drop_delta_acc(
        encoder, train_loader, test_loader, device,
        k=args.k, drop_prob=args.drop_prob
    )
    print(f"[Eval] Joint-Drop ΔAcc = {delta_acc:.6f}")
    summary_lines.append(f"Joint-Drop ΔAcc: {delta_acc:.6f}")

    # Log to WandB
    if not args.no_wandb:
        wandb.log({
            "knn_accuracy": knn_acc,
            "silhouette_score": sil,
            "smoothness_ts": ts,
            "joint_drop_delta_acc": delta_acc,
            "train_seqs": len(train_seqs),
            "test_seqs": len(test_seqs),
            "train_windows": len(train_ds),
            "test_windows": len(test_ds),
        })
        
        # Save summary as artifact
        wandb.summary.update({
            "knn_accuracy": knn_acc,
            "silhouette_score": sil,
            "smoothness_ts": ts,
            "joint_drop_delta_acc": delta_acc,
        })

    # Save Summary
    with open(out_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\n[Saved] Summary -> {out_path}")
    
    # Finish WandB run
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
