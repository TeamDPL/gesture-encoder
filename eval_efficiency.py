#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficiency Evaluation Script
Measures GFLOPs, #Params, and Throughput (FPS) for gesture encoders.
Output: result/efficiency/log_efficiency_YYMMDD_HHMMSS.txt
"""

import os
import time
import torch
import numpy as np
from datetime import datetime
from thop import profile
import tdgcn_dual
import tdgcn_dual_wrist
import stgcn_sl_64

def get_model(model_name, device):
    if model_name == "tdgcn_dual":
        return tdgcn_dual.get_encoder(device)
    elif model_name == "tdgcn_dual_wrist":
        return tdgcn_dual_wrist.get_encoder(device)
    elif model_name == "stgcn_sl_64":
        return stgcn_sl_64.get_encoder(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_flops(model, dummy_input):
    model.eval()
    # thop.profile expects inputs as *args
    if isinstance(dummy_input, (list, tuple)):
        flops, params = profile(model, inputs=dummy_input, verbose=False)
    else:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops / 1e9  # GFLOPs

def measure_throughput(model, dummy_input, num_batches=100, warmup=10):
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if isinstance(dummy_input, (list, tuple)):
                model(*dummy_input)
            else:
                model(dummy_input)
    
    # Measure
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_batches):
            if isinstance(dummy_input, (list, tuple)):
                model(*dummy_input)
            else:
                model(dummy_input)
                
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    total_time = end_time - start_time
    # FPS = (Batch Size * Num Batches) / Total Time
    # Assuming dummy_input batch size is 1 for throughput measurement per sample
    # Or we can measure batch throughput. Let's measure samples per second.
    
    if isinstance(dummy_input, (list, tuple)):
        batch_size = dummy_input[0].shape[0]
    else:
        batch_size = dummy_input.shape[0]
        
    fps = (batch_size * num_batches) / total_time
    return fps

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    models_to_test = ["stgcn_sl_64", "tdgcn_dual", "tdgcn_dual_wrist"]
    
    # Setup output
    out_dir = "result/efficiency"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"log_efficiency_{timestamp}.txt")
    
    results = []
    results.append(f"Efficiency Evaluation Report - {timestamp}")
    results.append(f"Device: {device}")
    results.append("-" * 50)
    results.append(f"{'Model':<20} | {'Params (M)':<10} | {'GFLOPs':<10} | {'FPS':<10}")
    results.append("-" * 50)
    
    print("-" * 50)
    print(f"{'Model':<20} | {'Params (M)':<10} | {'GFLOPs':<10} | {'FPS':<10}")
    print("-" * 50)
    
    for model_name in models_to_test:
        try:
            encoder = get_model(model_name, device)
            # Use the encoder wrapper directly to ensure forward signature matches
            model = encoder 
            
            # Prepare dummy input
            if model_name == "tdgcn_dual":
                # Input: (N, C, T, V, M) -> (2, 3, 64, 22, 1)
                dummy_input = torch.randn(2, 3, 64, 22, 1).to(device)
                
            elif model_name == "tdgcn_dual_wrist":
                # Input: (N, C, T, V, M) -> (2, 3, 64, 22, 1)
                # Aux removed
                dummy_input = torch.randn(2, 3, 64, 22, 1).to(device)
                
            elif model_name == "stgcn_sl_64":
                # Input: (N, C, T, V, M) -> (2, 3, 64, 27, 1)
                # Based on error "running_mean should contain 126 elements not 81"
                # The model weights have 81 channels (3*27), so V=27.
                dummy_input = torch.randn(2, 3, 64, 27, 1).to(device)
            
            # 1. Params
            params = count_params(model) / 1e6 # Million
            
            # 2. GFLOPs
            # For GFLOPs, we usually measure for a single sample (B=1, 1 hand or 2 hands?)
            # The model processes (B*2) hands.
            # Let's measure for processing ONE dual-hand sequence (so batch=2 effectively for the encoder)
            gflops = measure_flops(model, dummy_input)
            
            # 3. Throughput
            fps = measure_throughput(model, dummy_input)
            # Note: FPS here is "Dual-Hand Sequences per Second" because input batch is 2 (representing 1 seq of 2 hands)
            
            res_str = f"{model_name:<20} | {params:<10.2f} | {gflops:<10.4f} | {fps:<10.1f}"
            print(res_str)
            results.append(res_str)
            
        except Exception as e:
            err_str = f"{model_name:<20} | Error: {e}"
            print(err_str)
            results.append(err_str)
            
    with open(out_path, "w") as f:
        f.write("\n".join(results))
        
    print("-" * 50)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
