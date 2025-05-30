import json
import random
import os
from multiprocessing import Pool
from utils.utils import chat_completion
import random
from multiprocessing import Pool


save_path_dir = "data/RefineToolbench/Output/LLM_vllm_multi_agent_qwen2.5_sft1"
refine_toolbench_path = "/data/user/code/luckytool/data/RefineToolbench/data"
I1_data = json.load(open(os.path.join(refine_toolbench_path, "I1.json"), "r"))
I2_data = json.load(open(os.path.join(refine_toolbench_path, "I2.json"), "r"))
I3_data = json.load(open(os.path.join(refine_toolbench_path, "I3.json"), "r"))

check_is_correct_prompt = """
<function>
<name>parse_result_status</name>
Giving the query and some APIs, we have wrong actions, you need to check the after messages if the model is aware of the error and correct the error to get the correct final answer.
The result is in a json format like
```json
{
    "content": "the reason of the result status",
    "error_recognition": Pass or Fail,
    "error_correction": Pass or Fail
}
```
You should check by the following rules:
1. If the model recognizes there was an error in the API call or response, return Pass for error_recognition, otherwise return Fail.
2. If the model successfully corrects the error and gets valid results in subsequent API calls, return Pass for error_correction, otherwise return Fail.
3. The model must both recognize the error AND successfully correct it to get Pass for both fields.
4. If the model fails to recognize there was an error, both fields should be Fail.
5. If the model recognizes an error but fails to correct it, error_recognition should be Pass but error_correction should be Fail.
6. Only Pass or Fail values are allowed for both error_recognition and error_correction fields.
7. No other status values besides Pass/Fail should be used in the evaluation.

Start Messages (query and some APIs):
===start_message===

Wrong Action Messages (wrong api calling and api response):
===wrong_action_message===

The after messages (after the wrong action):
===after_message===

</function>
"""


def process_task(task):
    max_retry = 5
    for _ in range(max_retry):
        try:
            id, data, group, save_path = (
                task["id"],
                task["data"],
                task["group"],
                task["save_path"],
            )
            if data["win"] == False:
                data["result_status"] = {
                    "error_recognition": False,  # Whether the model recognized there was an error
                    "error_correction": False,  # Whether the model successfully corrected the error
                }
                return
            previous_messages = data["answer_generation"]["messages"][:2]
            wrong_message = data["answer_generation"]["messages"][2:4]
            correct_message = data["answer_generation"]["messages"][4:]
            prompt = (
                check_is_correct_prompt.replace(
                    "===start_message===", str(previous_messages)
                )
                .replace("===wrong_action_message===", str(wrong_message))
                .replace("===after_message===", str(correct_message))
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
                answer_status["error_recognition"],
                answer_status["error_correction"],
            )
            data["result_status"] = answer_status
            with open(save_path, "w") as writer:
                json.dump(data, writer, indent=2)
            break
        except Exception as e:
            print(e)
            continue


def process_task_I3(task):
    max_retry = 5
    for _ in range(max_retry):
        try:
            id, data, group, save_path = (
                task["id"],
                task["data"],
                task["group"],
                task["save_path"],
            )
            if data["win"] == False:
                data["result_status"] = {
                    "error_recognition": False,  # Whether the model recognized there was an error
                    "error_correction": False,  # Whether the model successfully corrected the error
                }
                return
            pre_messages_len = len(I3_data[int(id)]["pre_messages"])
            pre_messages = data["answer_generation"]["messages"][:pre_messages_len]
            wrong_message = data["answer_generation"]["messages"][
                pre_messages_len : pre_messages_len + 2
            ]
            correct_message = data["answer_generation"]["messages"][
                pre_messages_len + 2 :
            ]
            prompt = (
                check_is_correct_prompt.replace(
                    "===start_message===", str(pre_messages)
                )
                .replace("===wrong_action_message===", str(wrong_message))
                .replace("===after_message===", str(correct_message))
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
                answer_status["error_recognition"],
                answer_status["error_correction"],
            )
            data["result_status"] = answer_status
            with open(save_path, "w") as writer:
                json.dump(data, writer, indent=2)
            break
        except Exception as e:
            print(e)
            continue


all_task_I1_I2 = []
all_task_I3 = []
win_rate = {"I1": 0, "I2": 0, "I3": 0}
for filename in os.listdir(save_path_dir):
    file_path = os.path.join(save_path_dir, filename)
    with open(file_path, "r") as f:
        data = json.load(f)
    group, id = filename.split(".")[0].split("@")

    if data["win"] == True:
        win_rate[group] += 1
    else:
        continue
    if "result_status" in data:
        continue
    if group in ["I1", "I2"]:
        all_task_I1_I2.append(
            {"id": id, "data": data, "group": group, "save_path": file_path}
        )
    else:
        all_task_I3.append(
            {"id": id, "data": data, "group": group, "save_path": file_path}
        )
print(f"win rate I1: {win_rate['I1']/len(I1_data)}, {len(I1_data)}")
print(f"win rate I2: {win_rate['I2']/len(I2_data)}, {len(I2_data)}")
print(f"win rate I3: {win_rate['I3']/len(I3_data)}, {len(I3_data)}")
print(f"total task I1_I2: {len(all_task_I1_I2)}")
print(f"total task I3: {len(all_task_I3)}")


with Pool(processes=1) as pool:
    pool.map(process_task, all_task_I1_I2)

with Pool(processes=1) as pool:
    pool.map(process_task_I3, all_task_I3)
