import json
import os
from utils.utils import chat_completion
import random
from multiprocessing import Pool
import time
import tqdm

base_path = "stabletoolbench/data_eval/answer/new_train_0106"

all_task = []
for group in ["G3", "G2", "G1"]:
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
                        final_answer = data["answer_generation"]["final_answer"]
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


# prompt_template = """
# <function>
# <name>check_answer_status</name>
# <description>
# Giving the query and answer, you need give `answer_status` of the answer by following rules:
# 1. If the answer is a sorry message or not a positive/straight response for the given query, return "Unsolved".
# 2. If the answer is a positive/straight response for the given query, you have to further check.
# 2.1 If the answer is not sufficient to determine whether the solve the query or not, return "Unsure".
# 2.2 If you are confident that the answer is sufficient to determine whether the solve the query or not, return "Solved" or "Unsolved".

# Query:
# ===query===
# Answer:
# ===answer===

# Output your reason in "content" and `answer_status` of JSON to `check_answer_status`.
# The output format is:```json
# {
#     "content": "...",
#     "answer_status": "..."
# }```
# </description>
# </function>
# """

prompt_template = """
<function>
<name>check_answer_status_and_steps</name>
<description>
Giving the query and the corresponding reasoning process of an answer, you need to:
1. Determine the `answer_status` of the answer by following these rules:
   - If you are confident that the answer is sufficient to determine whether the query is solved or not, return "Pass" or "Fail".
   - If the answer is not sufficient to determine whether the query is solved or not, for example, you need to check the complete reasoning process and API response information, return "Unsure".
   - If there are tool executions in the chain containing successful function calls, and those calls indeed solve the query, return "Pass".
   - If the API does not provide valid information and the model has tried all APIs to retrieve useful information, but none of the APIs provide any, return "Unsure".
   - If all API Observation messages indicate errors occurred, return "Fail".
   - If you find the information in the "final_answer" is not true/valid according to the messages in API Observation, return "Fail".
   - If you are unable to verify the authenticity and validity of the information, return "Unsure".

2. Analyze the reasoning process to determine if there are any irrelevant steps or incorrect attempts that do not contribute to the final result:
   - If all steps in the reasoning process are meaningful and contribute to solving the query, set `all_steps_validity` to "yes".
   - If there are irrelevant or incorrect steps that do not contribute to solving the query, set `all_steps_validity` to "no".

Output your thought in "content" and include `answer_status` and `all_steps_validity` in the JSON output.
The output format is:
```json
{
    "content": "...",
    "answer_status": "...",
    "all_steps_validity": "..."
}
```


---

Query:
===query===
Reasoning Process and Answer:
===reasoning_process===

</description>
</function>
"""


def process_task(task):
    max_retry = 5
    for _ in range(max_retry):
        try:
            query = task["query"]
            answer = task["final_answer"]
            file_path = task["file_path"]
            execution_chain = task["execution_chain"]
            prompt = prompt_template.replace("===query===", query).replace(
                "===reasoning_process===", execution_chain
            )
            models = [
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-2024-05-13",
                "gpt-4o-2024-11-20",
                "gpt-4",
                "gpt-4-1106-preview",
            ]

            model = random.choice(models)
            if _ == 4:
                model = "gpt-4o-mini"
            result = chat_completion(
                "xxx",
                [{"role": "user", "content": prompt}],
                "baseurl",
                model=model,
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
                answer_status["all_steps_validity"],
                answer_status["content"][:60],
            )
            # 读取文件
            with open(file_path, "r") as f:
                data = json.load(f)
            data["answer_status"] = answer_status
            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            break
        except Exception as e:
            print(e)

            time.sleep(random.randint(1, 8))
            continue


if __name__ == "__main__":
    pool = Pool(processes=50)
    pool.map(process_task, all_task)
    pool.close()
    pool.join()
