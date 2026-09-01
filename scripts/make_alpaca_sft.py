"""从 HuggingFace 下载 Alpaca 数据集并转换为 sft.py 可用的 JSON 格式。

输出: data/sft_data_alpaca10k.json（默认 10000 条，可通过参数改）
格式: [{"instruction": "...", "input": "...", "response": "..."}]
- input 为空时省略 input 字段（保持与旧格式兼容）

运行:
    python scripts/make_alpaca_sft.py              # 默认 10000 条
    python scripts/make_alpaca_sft.py --n 52000 --out data/sft_data_alpaca_full.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000, help="取多少条")
    p.add_argument("--out", type=str,
                   default=os.path.join(ROOT, "data", "sft_data_alpaca10k.json"))
    args = p.parse_args()

    print("下载 Alpaca 数据集...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"总条数: {len(ds)}")

    records = []
    for i in range(min(args.n, len(ds))):
        ex = ds[i]
        rec = {
            "instruction": ex["instruction"].strip(),
            "input": ex["input"].strip() if ex["input"] else "",
            "response": ex["output"].strip(),
        }
        records.append(rec)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(records)} 条到 {args.out}")

    # 展示几条样例
    print("\n样例:")
    for r in records[:3]:
        print(f"  Q: {r['instruction'][:60]}")
        if r["input"]:
            print(f"  Input: {r['input'][:60]}")
        print(f"  A: {r['response'][:60]}")


if __name__ == "__main__":
    main()
