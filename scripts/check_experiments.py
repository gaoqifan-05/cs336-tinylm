"""核对所有实验目录的最佳 PPL（收尾用临时脚本）"""
import torch, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for d in sorted(glob.glob("checkpoints/run_*")):
    best_p = os.path.join(d, "best.pt")
    final_p = os.path.join(d, "final.pt")
    p = best_p if os.path.exists(best_p) else final_p
    ck = torch.load(p, map_location="cpu", weights_only=False)
    name = os.path.basename(d)
    # 提取 tag（run_时间戳_tag_...）
    parts = name.split("_")
    tag = parts[2] if len(parts) > 2 else name
    print(f"{name:60s} PPL={ck.get('val_ppl', 0):8.2f}  step={ck.get('step', 0):5d}  file={os.path.basename(p)}")
