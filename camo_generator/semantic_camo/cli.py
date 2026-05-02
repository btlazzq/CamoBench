import argparse
import asyncio
import json
import sys
from typing import List

try:
    from .pipeline import run_pipeline
    from .utils import dedup_keep_order
except ImportError as e:
    msg = (
        f"依赖缺失: {e}\n"
        "请先安装依赖，例如: pip install -r requirements.txt 或 pip install agentscope"
    )
    print(msg, file=sys.stderr, flush=True)
    sys.exit(1)



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--terms", type=str, default="", help="逗号分隔的term列表，例如 无人机,比大小游戏")
    p.add_argument("--terms_file", type=str, default="", help="每行一个term的txt文件")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--gen_num", type=int, default=10)
    p.add_argument("--repair_rounds", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="", help="输出目录，默认为 semantic_camo/output_baike")
    p.add_argument("--scene", type=str, default="", help="广搜场景限定，如 黑话、赌博 等，传给百科+AI网页搜索")
    return p.parse_args()



def load_terms(args) -> List[str]:
    terms: List[str] = []
    if args.terms.strip():
        terms.extend([t.strip() for t in args.terms.split(",") if t.strip()])
    if args.terms_file.strip():
        with open(args.terms_file, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                # 兼容「词\t标签」格式的输入，只取第1列作为真实 term
                if "\t" in raw:
                    term = raw.split("\t", 1)[0].strip()
                else:
                    term = raw
                if term:
                    terms.append(term)
    if not terms:
        terms = ["无人机", "遥控飞行器"]
    return dedup_keep_order(terms)



def main():
    args = parse_args()
    term_list = load_terms(args)
    print(f"正在处理 {len(term_list)} 个 term: {term_list}", flush=True)
    try:
        results, jsonl_path, xlsx_path, evidence_features_dir, rag_chunks_path = asyncio.run(
            run_pipeline(
                term_list,
                topk=args.topk,
                gen_num=args.gen_num,
                repair_rounds=args.repair_rounds,
                out_dir=args.out_dir or None,
                scene=args.scene.strip() or None,
            )
        )
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
        print("\n✅ 已输出：", flush=True)
        print("JSONL:", jsonl_path, flush=True)
        print("Excel:", xlsx_path, flush=True)
        print("证据+特征(JSON):", evidence_features_dir, flush=True)
        print("RAG chunks(按时间戳):", rag_chunks_path, flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"运行失败: {e}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
