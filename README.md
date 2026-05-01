# CamoBench: Understanding and Benchmarking Camouflaged Illicit Online Promotion in Chinese

This repository contains the **core code for CamoBench**, covering the full experimental pipeline for **canonical corpus construction, camouflaged text generation, black-box moderation system evaluation, general LLM evaluation, and robustness training for classification models**.  
It provides reproducible scripts and evaluation workflows, and does not include offensive exploitation tooling.

## Ethics & Usage Disclaimer
This repository is released for academic research purposes only.  
The dataset and code may contain examples of illicit, sensitive, or potentially harmful content (e.g., camouflaged illicit promotion). These examples are included strictly for the purpose of studying and improving content moderation systems.

By accessing, downloading, or using this repository, you agree to the following:
- ✅ Use is limited to lawful, ethical, and non-commercial research
- ❌ Any misuse is strictly prohibited, including generating or facilitating illegal, harmful, or unethical activities
- ⚖️ You must comply with all applicable local laws, regulations, and institutional policies
- 🚫 The authors shall not be held liable for any misuse or any consequences arising therefrom

This work may expose potential weaknesses in moderation systems. Such findings are shared responsibly and are intended only to advance safer and more robust AI systems.  
By downloading or using this repository, you acknowledge and agree to these terms.

## Repository Structure
- `canonical_builder/`: Canonical corpus construction (category-wise raw and normalized text generation).
- `camo_generator/`: Multi-strategy camouflaged text generation (abbreviation, emoji, phonetic, orthographic, semantic, etc.).
- `camo_evaluator/`: Model and moderation-system evaluation (sampling, inference, result splitting, and analysis).
- `robustness_trainer/`: Classification-model fine-tuning and robustness training (LoRA/DDP).
- `anchor_replace/`: Anchor-word replacement and coverage-balanced data augmentation scripts.
- `requirements.txt`: Python dependency list.
- `bash install.sh`: Conda environment bootstrap script.

## 1. Prerequisites

### Hardware & OS
* **OS**: Linux/macOS (Linux Ubuntu 22.04+ recommended).
* **Python**: Version 3.10 or higher.
* **GPU (optional but recommended)**: CUDA-capable GPU is recommended for training workloads.
* **Storage**: At least 30GB is recommended; required space grows with data scale and intermediate artifacts.

### External Services
To reproduce online evaluation workflows, you may need credentials for external APIs:
* **LLM API Keys**: such as OpenAI, Claude, Gemini, Doubao, DeepSeek, Grok, HunYuan, etc.
* **Content Moderation APIs**: such as Alibaba Cloud, Baidu, Tencent Cloud, NetEase Yidun, 360, depending on which evaluation scripts you run.

> Note: Different scripts read credentials from environment variables or local script configs. Prefer environment variables and avoid hardcoding secrets.

## 2. Installation & Setup

### Environment Setup
Clone the repository and install dependencies:

```bash
cd camobench
```

Create the environment with the bundled script (filename contains a space):

```bash
bash "bash install.sh"
```

Or set up manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### API Configuration
Set credentials based on the scripts you plan to run. For example, OpenAI-based evaluation scripts usually read:

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # optional
```

Other provider scripts accept similar environment variables or command-line arguments.

## 3. Experiment Workflow

### Part A: Canonical Corpus Construction (`canonical_builder`)
This part builds baseline corpora under category labels (e.g., pornographic, gambling, drugs, non-violative).

Main entry (example):
```bash
python -m canonical_builder.main
```

You can also invoke category-specific model API modules (examples):
```bash
python -m canonical_builder.model_api.ModelAPISeqing
python -m canonical_builder.model_api.ModelAPIDrugs
```

### Part B: Camo Text Generation (`camo_generator`)
This part generates multiple types of camouflaged expressions for robustness evaluation.

1) **Abbreviation camo**
```bash
python camo_generator/abbreviation_camo/batch_abbreviation_camo.py \
  --terms_file camo_generator/term_seqing.txt \
  --out_dir camo_generator/abbreviation_camo/output_abbreviation_porn
```

2) **Emoji camo**
```bash
python camo_generator/emoji_camo/batch_emoji_camo.py \
  --terms_file camo_generator/term_drugs.txt \
  --out_dir camo_generator/emoji_camo/output_emoji_drugs
```

3) **Phonetic camo**
```bash
python camo_generator/phonetic_camo/batch_phonetic_camo.py \
  --terms_file camo_generator/term_gambling.txt \
  --out_dir camo_generator/phonetic_camo/output_phonetic_gambling
```

4) **Orthographic camo**
```bash
python camo_generator/orthographic_camo/batch_orthographic_camo.py \
  --terms_file camo_generator/term_black.txt \
  --out_dir camo_generator/orthographic_camo/output_orthographic_black
```

5) **Semantic camo (retrieval + generation)**
```bash
python -m camo_generator.semantic_camo.cli \
  --terms_file camo_generator/term_none.txt \
  --topk 4 \
  --gen_num 8 \
  --repair_rounds 2 \
  --out_dir camo_generator/semantic_camo/output_none
```

### Part C: Model Evaluation (`camo_evaluator`)
This part evaluates how general LLMs or moderation systems identify camouflaged text.

1) **Run a model-specific evaluation script directly (ChatGPT example)**
```bash
python camo_evaluator/model_check/infer_eval_chatgpt.py \
  --input_jsonl data/eval.jsonl \
  --model gpt-4o \
  --pred_jsonl outputs/chatgpt_pred.jsonl
```

2) **Sample every 7 steps and aggregate metrics**
```bash
python camo_evaluator/sample_every7_and_eval.py \
  --gold_jsonl data/gold.jsonl \
  --pred_jsonl outputs/chatgpt_pred.jsonl \
  --out_dir outputs/eval_every7
```

3) **Split and summarize moderation-system outputs**
```bash
python camo_evaluator/system_result_split/text_system_check_aliyun_split.py
```

> `model_check/` also includes similar entries such as `infer_eval_claude.py`, `infer_eval_gemini.py`, `infer_eval_doubao.py`, and `infer_eval_deepseek.py`.

### Part D: Robustness Training (`robustness_trainer`)
LoRA-based classifier training (supports 4-bit quantization, DDP, etc.).

```bash
python robustness_trainer/train_qwen_cls_lora_ddp.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --train_file data/train.jsonl \
  --dev_file data/dev.jsonl \
  --output_dir outputs/qwen_cls_lora \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 4 \
  --use_bf16 \
  --use_4bit
```

### Part E: Anchor Replacement & Balance Runner (`anchor_replace`)
Used for anchor-term expansion and category-balanced sample construction.

```bash
python anchor_replace/word_camo_expand_porn.py
python anchor_replace/cover_balance_runner_porn.py
```

## 4. Suggested Reproduction Pipeline

Recommended minimal reproducible path:
1. Use `canonical_builder/` to generate or curate canonical corpora.
2. Use `camo_generator/` to produce multi-strategy camouflaged samples.
3. Use `camo_evaluator/model_check/` to run predictions on target models.
4. Use `sample_every7_and_eval.py` to generate grouped evaluation metrics.
5. (Optional) Train a defensive classifier in `robustness_trainer/` and evaluate again.

## Notes
* **API Limits**: Online API-based evaluation may hit rate limits (HTTP 429). Reduce concurrency, increase retries, or use higher-tier quotas.
* **Security**: This repository is intended for safety evaluation research and robustness experiments. Use only in legal, compliant, and authorized settings.
* **Credentials**: Never commit real API keys, credentials, tokens, or private data.
* **Reproducibility**: Fix random seeds, record model versions and key parameters, and preserve intermediate artifacts for auditing.
