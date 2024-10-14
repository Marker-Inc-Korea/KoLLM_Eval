# KoLMM_Eval🥰
한국어 벤치마크 평가 코드 통합본(?) **2024.10.15 Version**  
> Logickor, K2-Eval, and LM-Harness 평가를 하나의 코드에서 실행

# Install (required)🤩
First, download LM-Eval-Harness.  
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd ./lm-evaluation-harness
pip install -e .
pip install -e ".[multilingual]"

pip install vllm
```
  
Lastly, you need to move logickor and k2-eval codes.  
```
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


# References🌠
[Logickor](https://github.com/instructkr/LogicKor)  
[LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness)  
[K2-eval](https://huggingface.co/datasets/HAERAE-HUB/K2-Eval)   
