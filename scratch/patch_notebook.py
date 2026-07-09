import json
from pathlib import Path

notebook_path = Path("notebooks/percv_kaggle.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Update the torch.save cell
found_save = False
found_baseline = False

for cell in data["cells"]:
    if cell["cell_type"] != "code":
        continue
    
    source = cell["source"]
    source_str = "".join(source)
    
    if "torch.save(model.state_dict()" in source_str:
        # Re-split with proper newlines
        new_source = []
        for line in source:
            if "torch.save(model.state_dict()" in line:
                # Replace with rich saving dict
                new_source.extend([
                    "        torch.save({\n",
                    "            \"state_dict\": model.state_dict(),\n",
                    "            \"backbone\": backbone_type,\n",
                    "            \"class_names\": classes,\n",
                    "            \"input_size\": (128, 128),\n",
                    "            \"normalize_mean\": [0.485, 0.456, 0.406],\n",
                    "            \"normalize_std\": [0.229, 0.224, 0.225]\n",
                    "        }, f\"{exp_dir}/models/model_best.pt\")\n"
                ])
            else:
                new_source.append(line)
        cell["source"] = new_source
        found_save = True
        print("Updated torch.save cell in notebook.")
        
    if "baselines = {" in source_str and "mobilenetv2" in source_str:
        # Locate baselines structure and replace it
        # Let's completely rewrite the baselines section in the source lines
        new_source = []
        skip = False
        for line in source:
            if "baselines = {" in line:
                new_source.extend([
                    "    # Benchmark dashboard showing only backbones actually trained in this run\n",
                    "    baselines = {\n",
                    "        backbone_type: {\n",
                    "            \"params_m\": round(total_params / 1e6, 2),\n",
                    "            \"size_mb\": round(model_size_mb, 2),\n",
                    "            \"acc\": round(test_acc, 4),\n",
                    "            \"f1\": round(val_f1, 4),\n",
                    "            \"fps\": round(fps_rate, 1),\n",
                    "            \"train_time_sec\": round(t4_train_duration, 1)\n",
                    "        }\n",
                    "    }\n"
                ])
                skip = True
                continue
            if skip:
                # We skip lines until we reach the end of the baselines dict assignment
                # The baselines dict assignment ends before executed_key definition
                if "executed_key = backbone_type" in line:
                    skip = False
                    new_source.append(line)
                continue
            new_source.append(line)
            
        cell["source"] = new_source
        found_baseline = True
        print("Removed fabricated MobileNetV2 dict structure in notebook cell 18.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print(f"Finished patching. Save cell found: {found_save}, Baseline cell found: {found_baseline}")
