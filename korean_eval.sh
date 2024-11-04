## CSAT-QA eval
python lm_eval \
    --model hf \
    --model_args pretrained=HumanF-MarkrAI/Gukbap-Gemma2-9B \
    --tasks csatqa \
    --device cuda:0 \
    --batch_size 4 \
    --num_fewshot 0

## kmmlu_direct eval (5-shot)
python lm_eval \
    --model hf \
    --model_args pretrained=HumanF-MarkrAI/Gukbap-Gemma2-9B \
    --tasks kmmlu_direct \
    --device cuda:0 \
    --batch_size 4 \
    --num_fewshot 5

## haerae eval
python lm_eval \
    --model hf \
    --model_args pretrained=HumanF-MarkrAI/Gukbap-Gemma2-9B \
    --tasks haerae \
    --device cuda:0 \
    --batch_size 4 \
    --num_fewshot 0

## Logickor eval (2024.10.14 ver)
python ./MTBench/logickor.py \
    --is_multi_turn 1 \
    --eval_model gpt-4-1106-preview \
    --repo_name HumanF-MarkrAI \
    --base_model Gukbap-Gemma2-9B \
    --max_token 4096 \
    --huggingface_token '[...token...]' \
    --api '[...api...]'


## K2-eval
python ./MTBench/k2_eval.py \
    --is_multi_turn 0 \
    --eval_model gpt-4-1106-preview \
    --repo_name HumanF-MarkrAI \
    --base_model Gukbap-Gemma2-9B \
    --max_token 4096 \ 
    --huggingface_token '[...token...]' \
    --api '[...api...]'
