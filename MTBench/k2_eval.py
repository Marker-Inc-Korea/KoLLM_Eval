from vllm import LLM, SamplingParams
from tqdm import tqdm
from huggingface_hub import login
import pandas as pd
import json
import re
import argparse
import os
import time

from openai import OpenAI

####
'''
# k2_eval
Link: https://huggingface.co/datasets/HAERAE-HUB/K2-Eval?row=89
'''
####

if __name__ == '__main__':

    os.makedirs("./results", exist_ok=True)

    ## Argparse
    parser = argparse.ArgumentParser(description='config argparser')
    parser.add_argument('--is_multi_turn', type=int, default=0, help="0 is false, other is true")
    parser.add_argument('--eval_model', type=str, default='gpt-4-turbo', help='gpt-4-1106-preview, gpt-4-turbo, gpt-4o')
    parser.add_argument('--repo_name', type=str, default='MarkrAI', help="Huggingface repo name")
    parser.add_argument('--base_model', type=str, default='Ko-mistral-7B-Markr-Wizard-v2.4-epoch4')
    parser.add_argument('--max_token', type=int, default=8192)

    parser.add_argument('--api', type=str, required=True)
    parser.add_argument('--huggingface_token', type=str, required=True)
    args = parser.parse_args()

    args.is_multi_turn = bool(args.is_multi_turn)

    # CHECK flag
    print("###################################")
    print("Is multi_turn:", args.is_multi_turn)
    print("OpenAI eval model:", args.eval_model)
    print("Huggingface repo name:", args.repo_name)
    print("What is your base model:", args.base_model)
    print("What is the max token:", args.max_token)
    print("###################################")
    
    # Ref.: https://github.com/instructkr/LogicKor/blob/main/templates.py#L98
    JUDGE_TEMPLATE = {
        "single_turn": """너는 질문에 대한 한국어 언어 모델의 답변을 매우 꼼꼼히 평가할 것이다. 공정한 평가를 위해 아래의 규칙을 준수한다.
    
    # 기본 규칙
    1. 질문의 요구사항을 충분히 반영하였는지 상세히 분석할 것.
    2. 답변 과정에서 누락되었거나 포함되지 못하여 아쉬운 부분에 대하여 상세히 분석할 것.
    3. 답변의 길이가 평가 결과에 영향을 미치지 않도록 할 것.
    
    # 언어 요구사항
    - 모델은 반드시 한국어로 답변해야 하며, 다른 언어로의 답변은 절대 허용되지 않는다.
    - 예외적으로 질문이 영어로 답변할 것을 요구할 때에만 영어 답변이 허용된다.
    - 언어 요구사항을 충족하는 것은 필수적이나, 이 요구사항의 충족이 답변의 질적 평가에 추가 점수로 이어지지는 않는다.
    
    # 평가 출력 방식
    **주어진 Question에 집중하여** Model's Response에 대한 평가와 1~5의 점수를 부여한다. 답변에 대한 평가는 4~5 문장으로 규칙을 참고하여 상세히 작성한다.
    
    # 출력 형식
    평가: 평가 내용
    점수: 숫자""",
        
        "multi_turn": "" # not consider
    }

    # Setting
    login(token=args.huggingface_token)

    # OpenAi
    client = OpenAI(api_key=args.api)

    # Model define
    max_token = args.max_token
    model_name = args.base_model
    base_model = args.repo_name + '/' + model_name
    
    # 'MarkrAI/Ko-mistral-7B-Markr-Wizard-v2.4-epoch4'
    # 'Ko-mistral-7B-Markr-Wizard-v2.2-epoch5'
    model = LLM(model=base_model, tensor_parallel_size=2, max_model_len=max_token, gpu_memory_utilization=0.95, swap_space=16, trust_remote_code=True)
    
    sampling_params = SamplingParams(temperature=0.0, # 0.9
                                    skip_special_tokens=True,
                                    max_tokens=max_token,
                                    stop=["<|endoftext|>", "[INST]", "[/INST]", "<|im_end|>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<eos>"])

    # Loading data
    print("Loading data")
    df_questions = pd.read_csv("data_k2-eval-generation.csv")
    print(df_questions)

    def format_single_turn_question(question):
        return model.llm_engine.tokenizer.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    single_turn_questions = df_questions["instruction"].map(format_single_turn_question)

    # Model output
    iteration = 1
    is_multi_turn = args.is_multi_turn
    results = os.listdir("./results")

    file_name = "./results/K2_Eval_" + model_name+"_0.jsonl"
    
    if not file_name in results:
        for i in range(iteration):
            single_turn_outputs = [
                output.outputs[0].text.strip() for output in tqdm(model.generate(single_turn_questions, sampling_params))
            ]
    
            # multi turn generator
            if is_multi_turn:
                def format_double_turn_question(question, single_turn_output):
                    return model.llm_engine.tokenizer.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": question[0]},
                            {"role": "assistant", "content": single_turn_output},
                            {"role": "user", "content": question[1]},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
    
                multi_turn_questions = df_questions[["instruction", "subject"]].apply(
                    lambda x: format_double_turn_question(x["instruction"], single_turn_outputs[x["subject"] - 1]),
                    axis=1,
                )
                multi_turn_outputs = [
                    output.outputs[0].text.strip() for output in tqdm(model.generate(multi_turn_questions, sampling_params))
                ]
    
                # saving
                df_output = pd.DataFrame(
                    {
                        "subject": df_questions["subject"],
                        "instruction": df_questions["instruction"],
                        "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
                    }
                )
    
            else:
    
                # saving
                df_output = pd.DataFrame(
                    {
                        "subject": df_questions["subject"],
                        "instruction": df_questions["instruction"],
                        "outputs": list(zip(single_turn_outputs, )),
                    }
                )
    
            #try:
            #    df_output.to_excel('./results/'+model_name+"_"+str(i)+".xlsx", index=False)
            #except:
            df_output.to_json(
                            "./results/K2_Eval_" + model_name+"_"+str(i)+".jsonl",
                            orient="records",
                            lines=True,
                            force_ascii=False,
                            )
            #num += 1
    
    else:
        print("Already finished")
    
    
    ###### Evaluation
    eval_model = args.eval_model # gpt-4-turbo, gpt-4o, gpt-4-1106-preview
    score_iteration = 2
    for i in range(score_iteration):
        df_generated = pd.read_json("./results/K2_Eval_"+model_name+"_"+str(i)+".jsonl", orient="records", encoding="utf-8-sig", lines=True)

        #print(df_generated)

        score_list = []
        judge_list = []
        multi_score_list = []
        multi_judge_list = []
    
        # Make prompt
        for k in tqdm(range(len(df_generated))):
            model_questions = df_generated.iloc[k, 1]
            model_outputs = df_generated.iloc[k, 2]
            
            #####################
            prompt = (
                f"아래의 내용을 주어진 평가 기준들을 충실히 반영하여 평가해라. 특히 모델 답변이 언어 요구사항을 준수하는지 반드시 확인해야 한다.\n\n"
                f"**Question**\n{model_questions}"
            )
        
            #if model_references and model_references[0]:
            #    prompt += f"\n\n**Additional Reference**\n{model_references[0]}"
            
            prompt += f"\n\n**Model's Response**\n{model_outputs[0]}"
        
            prompt += "\n\n[[대화 종료. 평가 시작.]]"
            #####################

            if k == 0:
                print(prompt)
        
            # Model output
            count = 0
            flag_again = True
            while flag_again:
                try:
                    response = client.chat.completions.create(
                                    model=eval_model, # gpt-4-turbo, gpt-4o, gpt-4-1106-preview
                                    temperature=0.0,
                                    n=1,
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": JUDGE_TEMPLATE["single_turn"],
                                        },
                                        {"role": "user", "content": prompt},
                                    ],
                                )

                    content = response.choices[0].message.content
                    judge_message_match = re.search(r"평가:(.*?)점수:", content.replace("*", ""), re.DOTALL)
                    judge_message = judge_message_match.group(1).strip() if judge_message_match else "No judge message found"
                    judge_score_match = re.search(r"점수:\s*(\d+(\.\d+)?)", content.replace("*", ""))
                    if judge_score_match:
                        judge_score = float(judge_score_match.group(1))
                    else:
                        raise ValueError("No score found in response")

                    flag_again = False
                    
                except Exception as E:
                    print(E)
                    count += 1
                
                    if count == 3:
                        judge_score = 0.0
                        judge_message = "Impossible to judge due to repetition."
                        flag_again = False
                    else:
                        print("Try after 20 sec...")
                        time.sleep(20)
        
            #judge_dict = {"judge_message": judge_message, "judge_score": judge_score}
            #print(judge_dict)
            score_list.append(judge_score)
            judge_list.append(judge_message)

            # multi_turn evaluator
            if is_multi_turn:

                #####################
                prompt = (
                    f"아래의 내용을 주어진 평가 기준들을 충실히 반영하여 평가해라. 특히 모델 답변이 언어 요구사항을 준수하는지 반드시 확인해야 한다.\n\n"
                    f"**Question**\n{model_questions[0]}"
                )
            
                if model_references and model_references[0]:
                    prompt += f"\n\n**Additional Reference**\n{model_references[0]}"
                
                prompt += f"\n\n**Model's Response**\n{model_outputs[0]}"

                # second turn
                prompt += f"\n\n**Follow-up Question.**\n{model_questions[1]}"
                
                if model_references and model_references[1]:
                    prompt += f"\n\n**Additional Reference**\n{model_references[1]}"
                    
                prompt += f"\n\n**Model's Response**\n{model_outputs[1]}"

                # end
                prompt += "\n\n[[대화 종료. 평가 시작.]]"
                #####################

                if k == 0:
                    print(prompt)

                # Model output
                count = 0
                flag_again = True
                while flag_again:
                    try:
                        response = client.chat.completions.create(
                                        model=eval_model, # gpt-4-turbo, gpt-4o, gpt-4-1106-preview
                                        temperature=0.0,
                                        n=1,
                                        messages=[
                                            {
                                                "role": "system",
                                                "content": JUDGE_TEMPLATE["multi_turn"],
                                            },
                                            {"role": "user", "content": prompt},
                                        ],
                                    )
                        
                        content = response.choices[0].message.content
                        judge_message_match = re.search(r"평가:(.*?)점수:", content.replace("*", ""), re.DOTALL)
                        judge_message = judge_message_match.group(1).strip() if judge_message_match else "No judge message found"
                        judge_score_match = re.search(r"점수:\s*(\d+(\.\d+)?)", content.replace("*", ""))
                        if judge_score_match:
                            judge_score = float(judge_score_match.group(1))
                        else:
                            raise ValueError("No score found in response")\
    
                        flag_again = False

                    except Exception as E:
                        print(E)
                        count += 1
                    
                        if count == 3:
                            judge_score = 0.0
                            judge_message = "Impossible to judge due to repetition."
                            flag_again = False
                        else:
                            print("Try after 20 sec...")
                            time.sleep(20)
            
                #judge_dict = {"judge_message": judge_message, "judge_score": judge_score}
                #print(judge_dict)
                multi_score_list.append(judge_score)
                multi_judge_list.append(judge_message)

        # mean score
        single_score = sum(score_list)/len(score_list)
        print("Single Average score:", sum(score_list)/len(score_list))
        
        if is_multi_turn:
            multi_score = sum(multi_score_list)/len(multi_score_list)
            print("Multi Average score:", sum(multi_score_list)/len(multi_score_list))
            print("All Average score:", (single_score+multi_score)/2)

            ## saving
            df_output = pd.DataFrame(
                {
                    "subject": df_questions["subject"],
                    "questions": df_questions["questions"],
                    "single_outputs": list(single_turn_outputs),
                    "references": df_questions["references"],
                    "single_judge_message": judge_list,
                    "single_judge_score": score_list,
                    "multi_outputs": list(multi_turn_outputs),
                    "multi_judge_message": multi_judge_list,
                    "multi_judge_score": multi_score_list,
                }
            )
        else:
            ## saving
            df_output = pd.DataFrame(
                {
                    "subject": df_questions["subject"],
                    "instruction": df_questions["instruction"],
                    "outputs": list(single_turn_outputs),
                    "single_judge_message": judge_list,
                    "single_judge_score": score_list,
                }
            )

        try:
            df_output.to_excel('./results/K2_Eval_'+model_name+"_"+str(i)+".xlsx", index=False)
        except:
            df_output.to_json(
                            "./results/K2_Eval_" + model_name+"_"+str(i)+".jsonl",
                            orient="records",
                            lines=True,
                            force_ascii=False,
                            )
