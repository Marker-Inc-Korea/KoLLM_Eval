export OPENAI_API_KEY='[...api...]'

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

#'''
########## KoMT-Bench scripts (run_ko_mt.sh)
cd ./KoMT-Bench/FastChat/fastchat/llm_judge/
# Generating model answers

CUDA_VISIBLE_DEVICES=0 python gen_model_answer.py \
		--model-path HumanF-MarkrAI/Gukbap-Gemma2-9B \
		--model-id Gukbap-Gemma2-9B \
		--dtype float16 \
	        --max-new-token 4096


# Assessing the model answer through LLM-as-a-judge (here, "gpt-4-0613")
python gen_judgment.py \
    --model-list Gukbap-Gemma2-9B


# Giving a penalty to the score of non-Korean responses
cd ./data/mt_bench/model_judgment
python detector.py \
    --model_id Gukbap-Gemma2-9B


# Showing the evaluation results
cd ../../..
python show_result.py \
    --mode single \
    --input-file ./data/mt_bench/model_judgment/Gukbap-Gemma2-9B_single_final.jsonl

cd ../../
#'''
