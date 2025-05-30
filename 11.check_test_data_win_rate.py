import json
import os
from utils.utils import chat_completion
import random
from multiprocessing import Pool
import time
import argparse
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
# base_path = "stabletoolbench/data_eval/answer/0109_test_lucky_tool_sft1_llama3.1_20250108_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0109_test_lucky_tool_sft1_qwen2.5_20250108_multi_agent"
base_path = "stabletoolbench/data_eval/answer/0114_test_lucky_tool_sft12_llama3.1_20250113_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0108_test_qwen2.5_sft_DFSFT"
# base_path = "stabletoolbench/data_eval/answer/0104_test_llama3.1_instruct_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0104_test_qwen2.5_instruct_multi_agent"

# reference_path = "stabletoolbench/data_eval/answer/0108_test_gpt3.5_instruct_DFSDT"
reference_path = "stabletoolbench/data_eval/answer/0108_test_llama3.1_sft_DFSFT"


prompt2_template = """

Query:
{query}

Answer_0:
{answer_0}

Answer_1:
{answer_1}

Given above query and answers in JSON format, you must follow the rules to select the relatively better answer and give the index of the answer **(0 for Answer_0, 1 for Answer_1)**:

1. Compare the quality of final answer (60% weight):
- Completeness (20%): Whether it contains ALL required information to fully answer the query
- Accuracy (20%): Whether all information provided is factually correct and matches the API responses
- Clarity (10%): Whether the answer is well-structured and easy to understand
- Error Handling (10%): If failed, whether it clearly explains what went wrong and why

2. Compare the execution efficiency (40% weight):
- Tool Usage (15%):
  * Minimize failed API calls (each failed call -2 points)
  * Avoid redundant API calls with same parameters (-1 point each)
  * Use appropriate tools that match the query needs
- Execution Path (15%):
  * Follow logical steps to solve the task
  * Reach key milestones in minimal steps
  * Stop when goal is achieved without unnecessary calls
- Strategy (10%):
  * Show good planning in tool selection
  * Adapt strategy based on API responses
  * Balance between exploration and exploitation

Select the answer with higher total score after weighted evaluation. In case of a tie, prefer the one with better final answer quality.

Output your reason in "content" and `better_answer_index` of JSON. better_answer_index must be 0 or 1, not other value.
The output format is:
```json
{
    "content": "thought process",
    "better_answer_index": "0 or 1"
}
```
"""


def process_task(task):
    max_retry = 5
    for _ in range(max_retry):
        try:
            query = task["query"]
            execution_chain_0 = task["execution_chain_0"]
            execution_chain_1 = task["execution_chain_1"]
            file_path = task["file_path"]
            prompt = (
                prompt2_template.replace("===query===", query)
                .replace("===answer_0===", execution_chain_0)
                .replace("===answer_1===", execution_chain_1)
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
            better_answer_index = str(
                json.loads(content)["better_answer_index"]
            ).strip()
            better_answer_index = int(better_answer_index)
            reason = json.loads(content)["content"]
            print(task["id"], better_answer_index, reason[:120])

            with open(file_path, "r") as f:
                data = json.load(f)
            if better_answer_index == 0:
                data["win_rate_status_toolllama3.1"] = "Pass"
            elif better_answer_index == 1:
                data["win_rate_status_toolllama3.1"] = "Fail"
            else:
                raise Exception("better_answer_index is not 0 or 1")
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            break
        except Exception as e:
            print(e)
            time.sleep(random.randint(1, 3))
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check test data win rate")
    # parser.add_argument('--base_path', type=str, required=True,
    #                   help='Base path for the test data')
    # args = parser.parse_args()
    # base_path = args.base_path
    print(base_path)
    all_task = []
    for group in [
        "G1_instruction",
        "G1_category",
        "G1_tool",
        "G2_category",
        "G2_instruction",
        "G3_instruction",
    ]:
        folder_path = os.path.join(base_path, group)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        # if "win_rate_status_toolllama3.1" in data:
                        #     continue
                        is_reference_pass = 0
                        is_predict_pass = 0
                        id = filename.split(".")[0]
                        id = id.split("_")[0]
                        if os.path.exists(
                            os.path.join(
                                reference_path, group, id + "_DFS_woFilter_w2.json"
                            )
                        ):
                            with open(
                                os.path.join(
                                    reference_path, group, id + "_DFS_woFilter_w2.json"
                                ),
                                "r",
                            ) as f:
                                data_reference = json.load(f)
                                if data_reference["win"] is False:
                                    is_reference_pass = 0
                                if data_reference["win"]:
                                    if (
                                        data_reference["answer_status"]["answer_status"]
                                        == "Pass"
                                        or data_reference["answer_status"][
                                            "answer_status"
                                        ]
                                        == "true"
                                        or data_reference["answer_status"][
                                            "answer_status"
                                        ]
                                        == True
                                    ):
                                        is_reference_pass = 2
                                    elif (
                                        data_reference["answer_status"]["answer_status"]
                                        == "Unsure"
                                    ):
                                        is_reference_pass = 1
                                    else:
                                        is_reference_pass = 0
                                if data["win"] is False:
                                    is_predict_pass = 0
                                elif "answer_status" not in data:
                                    is_predict_pass = is_reference_pass
                                else:
                                    if (
                                        data["answer_status"]["answer_status"] == "Pass"
                                        or data["answer_status"]["answer_status"]
                                        == "true"
                                        or data["answer_status"]["answer_status"]
                                        == True
                                    ):
                                        is_predict_pass = 2
                                    elif (
                                        data["answer_status"]["answer_status"]
                                        == "Unsure"
                                    ):
                                        is_predict_pass = 1
                                    else:
                                        is_predict_pass = 0

                                if is_predict_pass > is_reference_pass:
                                    data["win_rate_status_toolllama3.1"] = "Pass"
                                elif is_predict_pass < is_reference_pass:
                                    data["win_rate_status_toolllama3.1"] = "Fail"
                                else:

                                    if "actions" in data["answer_generation"]:
                                        execution_chain_0 = str(
                                            data["answer_generation"]["actions"]
                                        )
                                    else:
                                        execution_chain_0 = str(data["tree"])
                                    if "actions" in data_reference["answer_generation"]:
                                        execution_chain_1 = str(
                                            data_reference["answer_generation"][
                                                "actions"
                                            ]
                                        )
                                    else:
                                        execution_chain_1 = str(data_reference["tree"])
                                    query = data["answer_generation"]["query"]
                                    all_task.append(
                                        {
                                            "id": str(group) + "@" + id,
                                            "query": query,
                                            "execution_chain_0": execution_chain_0,
                                            "execution_chain_1": execution_chain_1,
                                            "file_path": file_path,
                                        }
                                    )
                        if "win_rate_status_toolllama3.1" in data:
                            with open(file_path, "w") as f:
                                json.dump(data, f, indent=4)
    print(len(all_task))
    pool = Pool(processes=50)
    pool.map(process_task, all_task)
    pool.close()
    pool.join()
