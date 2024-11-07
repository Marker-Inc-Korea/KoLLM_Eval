# KoLLM_Eval🥰
한국어 벤치마크 평가 코드 통합본(?) **2024.11.09 Version**  
> Logickor, K2-Eval, LM-Harness and KoMT-Bench 평가를 하나의 코드에서 실행
  
**🍚[Gukbap-Series LLM](https://huggingface.co/collections/HumanF-MarkrAI/gukbap-series-llm-66d32e5e8da15c515181b071)🍚**
  
# Install (required)🤩
First, download **⭐LM-Eval-Harness**.  
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd ./lm-evaluation-harness
pip install -e .
pip install -e ".[multilingual]"

pip install vllm
```
  
Secondly, download **⭐KoMT-Bench**.
```bash
(we will add)
```
  
Lastly, you need to move below `files`.  
```
questions.jsonl # logickor
data_k2-eval-generation.csv # k2_eval

korean_eval.sh
MTBench 
├──logickor.py
└──k2_eval.py
```

# Implementation🤩
```bash
sh korean_eval.sh
```
You must set `api key` through OpenAI.  
> You can test on a A100 GPU (using COLAB).

# Examples🤩
| Model | Logickor(0-shot) | K^2-Eval | Haerae(Acc) | CSAT-QA(Acc) | kmmlu(Acc) |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| `Human-MarkrAI/Gukbap-Gemma2-9B` | **8.77** | **4.50** | 62.60 | 43.85 | **46.46** |
| `google/gemma-2-9b-it` | 8.32 | 4.38 | 64.34 | 47.06 | 42.51 | 
| `rtzr/ko-geamm-2-9b-it` | 8.67 | 4.40 | 64.07 | **48.13** | 44.75 |
| `LGAI/EXAONE-3.0-7.8B-Instruct` | 8.64 | 4.43 | **77.09** | 34.76 | 35.23 |
| `yanolja/EEVE-Korean-Instruct-10.8B-v1.0` | 6.03 | 3.51 | 70.94 | 38.50 | 41.99 |
> Evaluator: GPT-4-1106-preview
> Logickor [0,10] & K^2-Eval [0,5]

# References🌠
[Logickor](https://github.com/instructkr/LogicKor)  
[LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness)  
[K2-eval](https://huggingface.co/datasets/HAERAE-HUB/K2-Eval)   
[KoMT-Bench](https://github.com/LG-AI-EXAONE/KoMT-Bench/tree/main)  
