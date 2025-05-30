import json
import os
from utils.utils import chat_completion
import random
from multiprocessing import Pool
import time
import tqdm

# base_path = "stabletoolbench/data_eval/answer/1231_test_lucky_tool_sft1_qwen2.5_20241230"
# base_path = "stabletoolbench/data_eval/answer/0101_test_lucky_tool_sft1_qwen2.5_20241231"
# base_path = "stabletoolbench/data_eval/answer/0101_test_lucky_tool_sft1_qwen2.5_20241231"
# base_path = "stabletoolbench/data_eval/answer/0101_test_lucky_tool_qwen2.5_instruct"
# base_path = "stabletoolbench/data_eval/answer/0103_test_lucky_tool_sft1_qwen2.5_20241231"
# base_path = "stabletoolbench/data_eval/answer/0103_test_lucky_tool_sft1_qwen2.5_20250102"
# base_path = "stabletoolbench/data_eval/answer/0104_test_qwen2.5_instruct_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0104_test_gpt4_instruct_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0104_test_gpt3.5_instruct_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0104_test_llama3.1_instruct_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0109_test_lucky_tool_sft1_qwen2.5_20250108_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0109_test_lucky_tool_sft1_llama3.1_20250108_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0112_test_lucky_tool_sft12_qwen2.5_20250111_multi_agent"
base_path = "stabletoolbench/data_eval/answer/0114_test_lucky_tool_sft12_qwen2.5_20250113_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0112_test_lucky_tool_sft12_llama3.1_20250111_multi_agent"

# "G2_category", "G2_instruction", "G3_instruction"
for group in [
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_category",
    "G2_instruction",
    "G3_instruction",
]:
    # for group in ["G2_category", "G2_instruction", "G3_instruction"]:
    folder_path = os.path.join(base_path, group)
    if os.path.exists(folder_path):
        all_cnt = len(os.listdir(folder_path))
        solved_cnt = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r") as f:
                    # print(file_path)
                    data = json.load(f)
                    if data.get("win"):
                        solved_cnt += 1
        print("Success rate", group, solved_cnt, all_cnt, solved_cnt / all_cnt)


all_task = []
for group in [
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_category",
    "G2_instruction",
    "G3_instruction",
]:
    # for group in [ "G2_category", "G2_instruction", "G3_instruction"]:
    folder_path = os.path.join(base_path, group)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if "answer_status" in data:
                        continue
                    if data.get("win") == True:
                        id = filename.split(".")[0]
                        id = id.split("_")[0]
                        # final_answer = data["answer_generation"]["final_answer"]
                        final_answer = data["answer_generation"]["messages"][-1][
                            "content"
                        ]
                        execution_chain = str(data["answer_generation"]["actions"])
                        query = data["answer_generation"]["query"]
                        if len(final_answer) > 0:
                            all_task.append(
                                {
                                    "id": str(group) + "@" + id,
                                    "query": query,
                                    "execution_chain": execution_chain,
                                    "final_answer": final_answer,
                                    "file_path": file_path,
                                }
                            )
print(len(all_task))


prompt_template = """
<function>
<name>check_answer_status</name>
<description>
Giving the query and answer, which the answer is provided by a LLM agent with API calls, you need give `answer_status` of the answer by following rules:
1. If you are confident that the answer is sufficient to determine whether the solve the query or not, return "Pass" or "Fail".
2. If the answer is not sufficient to determine whether the solve the query or not, for example you need to check the complete reasoning process and API response information, return "Unsure".

Query:
===query===
Answer:
===answer===

Output your reason in "content" and `answer_status` of JSON to `check_answer_status`.
The output format is:```json
{
    "content": "...",
    "answer_status": "..."
}```
</description>
</function>
"""

prompt2_template = """
<function>
<name>parse_answer_status</name>
<description>
Giving the query and the correspond execution detail of an answer, you need give `answer_status` of the answer by following rules:
1. If you are confident that the answer is sufficient to determine whether the solve the query or not, return "Pass" or "Fail".
2. If the answer is not sufficient to determine whether the solve the query or not, for example you need to check the complete reasoning process and API response information.
3. If there are tool execution in the chain contains successful func calling and those calling indeed solve the query, return "Pass".
4. If the answer is not correct at the beginning, but the model corrects the error in the subsequent reasoning and provides the correct answer through accurate API calls, return "Pass".
5. If API does not provide valid information and the model has tried all APIs to retrieve useful information, but the API does not provide any useful information, return "Unsure".
6. If all API Observation messages indicate that there are errors happened, return "Failed".
7. If you find the information in the "final_answer" is not true/valid according to the messages in API Observation, return "Failed".
8. If you are unable to verify the authenticity and validity of the information, return "Unsure"

Query:
===query===
Answer:
===answer===

Output your reason in "content" and `answer_status` of JSON.
The output format is:```json
{
    "content": "...",
    "answer_status": "..."
}```
</description>
</function>
"""


def process_task(task):
    max_retry = 5
    for _ in range(max_retry):
        try:
            query = task["query"]
            answer = task["final_answer"]
            execution_chain = task["execution_chain"]
            file_path = task["file_path"]
            prompt = prompt2_template.replace("===query===", query).replace(
                "===answer===", execution_chain
            )
            model = "gpt-4-turbo"

            result = chat_completion(
                "xxx",
                [{"role": "user", "content": prompt}],
                "baseurl",
                model=model,
                temperature=0,
            )
            content = result["choices"][0]["message"]["content"]
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            else:
                content = content.strip("{}")
                content = "{" + content + "}"
            answer_status = json.loads(content)

            print(
                task["id"],
                answer_status["answer_status"],
                answer_status["content"][:100],
            )
            # if answer_status["answer_status"] == "Unsure":
            #     prompt = prompt2_template.replace("===query===", query).replace("===answer===", execution_chain)
            #     result = chat_completion("xxx", [{"role": "user", "content":prompt}],"baseurl",model=model)
            #     content = result["choices"][0]["message"]["content"]
            #     if "```json" in content:
            #         content = content.split("```json")[1].split("```")[0].strip()
            #     else:
            #         content = content.strip("{}")
            #         content = "{" + content + "}"
            #     answer_status = json.loads(content)
            # 读取文件
            with open(file_path, "r") as f:
                data = json.load(f)
            data["answer_status"] = answer_status
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            break
        except Exception as e:
            print(e)

            time.sleep(random.randint(1, 3))
            continue


if __name__ == "__main__":
    pool = Pool(processes=30)
    pool.map(process_task, all_task)
    pool.close()
    pool.join()
