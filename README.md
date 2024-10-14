# KoLMM_Eval🥰
한국어 벤치마크 평가 코드 통합본(?) **2024.10.15 Version**  
> Logickor, K2-Eval, LM-Harness 평가를 하나의 코드에서 실행

# Install (required)🤩
```bash
git clone https://github.com/Marker-Inc-Korea/KoLMM_Eval
cd ./KoLMM_Eval
pip install -e .
pip install -e ".[multilingual]"

pip install vllm
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
