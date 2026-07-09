"""
Checkpoint inspector script — runs inside the backend container (or anywhere with torch).
Answers EXACTLY the questions in Part A checklist:
1. What type is the checkpoint?
2. Are there module. prefix keys?
3. Is there a saved class_to_idx?
4. What are the first few key names?
"""
import sys
import torch
from pathlib import Path

CANDIDATES = [
    Path("/app/model_best.pt"),
    Path("/app/artifacts/model_best.pt"),
    Path("model_best.pt"),
    Path("artifacts/model_best.pt"),
    Path("C:/Users/ahmad/Desktop/PerCV/model_best.pt"),
]

ckpt_path = None
for p in CANDIDATES:
    if p.exists():
        ckpt_path = p
        break

if ckpt_path is None:
    print("ERROR: Could not find model_best.pt in any candidate location.")
    sys.exit(1)

print(f"\n=== Checkpoint: {ckpt_path} ===")
ckpt = torch.load(ckpt_path, map_location="cpu")

print(f"\n[CHECK 1] Type: {type(ckpt).__name__}")

if isinstance(ckpt, dict):
    print(f"[CHECK 1] Dict keys: {list(ckpt.keys())}")
    
    # Locate the state_dict
    state_dict = None
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("[CHECK 1] Checkpoint format: WRAPPER DICT with 'model_state_dict' key")
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("[CHECK 1] Checkpoint format: WRAPPER DICT with 'state_dict' key")
    else:
        # Assume the dict itself is the state_dict
        state_dict = ckpt
        print("[CHECK 1] Checkpoint format: RAW STATE_DICT (no wrapper)")
    
    # Check for class_to_idx
    if "class_to_idx" in ckpt:
        print(f"\n[CHECK 3] class_to_idx FOUND in checkpoint: {ckpt['class_to_idx']}")
    else:
        print("\n[CHECK 3] class_to_idx NOT found in checkpoint.")
    
    # Check for other metadata keys
    meta_keys = [k for k in ckpt.keys() if k not in ("model_state_dict", "state_dict")]
    if meta_keys:
        print(f"[INFO] Other metadata keys: {meta_keys}")
        for k in meta_keys:
            v = ckpt[k]
            print(f"  {k}: {repr(v)[:300]}")
    
    # Check for module. prefix
    first_keys = list(state_dict.keys())[:10]
    print(f"\n[CHECK 2] First 10 state_dict keys:")
    for k in first_keys:
        print(f"  {k}")
    has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
    print(f"\n[CHECK 2] 'module.' prefix present: {has_module_prefix}")
    
    print(f"\n[INFO] Total state_dict keys: {len(state_dict)}")
    
elif isinstance(ckpt, torch.nn.Module):
    print("[CHECK 1] Checkpoint is a pickled Module object (not state_dict)")
else:
    print(f"[CHECK 1] Unknown checkpoint type: {type(ckpt)}")

print("\n=== Done ===")
