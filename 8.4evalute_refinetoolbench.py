import json
import random
import os
from toolbench.inference.Prompts.MultiAgent_prompts import (
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_WITHOUT_TOOL,
    FORMAT_INSTRUCTIONS_USER_FUNCTION,
)
from toolbench.inference.Algorithms.multi_agent import multi_agent

from toolbench.inference.LLM.llm_model import LLM
from toolbench.inference.LLM.llm_vllm import LLM_vllm
from multiprocessing import Pool


def replace_system_prompt(prompt):
    example_prompt = """### Example:
Task:
On Monday, September 30, 2024, I want to travel to Shanghai. How much is the cheapest Hilton hotel for one night? Also, how much is the cheapest ticket for Shanghai Disneyland for the next day, October 1, 2024?

Assistant [1]:
 <thought>
First, I need to query the price of the cheapest Hilton hotel in Shanghai on September 30, 2024. I will use the `query_hotel_list_by_demand` API and set `city` to "Shanghai" and `demand` to "Hilton Hotel".
</thought>
<execute>
```python
print(query_hotel_list_by_demand(city="Shanghai", demand="Hilton Hotel"))
```
</execute>

Observation [1]:
The cheapest Hilton hotel is the Shanghai Hongqiao National Exhibition Center Hilton Hampton Hotel, with a price starting at ￥557.

Assistant [2]:
 <thought>
Next, I need to query the price of a ticket to Shanghai Disneyland for October 1, 2024. I will use the `search_scenic_spot_ticket_info` API with the query "Shanghai Disneyland ticket price on October 1, 2024" and set `city` to "Shanghai".
</thought>
<execute>
```python
print(search_scenic_spot_ticket_info(query="2024-10-01 Shanghai Disneyland ticket price", city="Shanghai"))
```
</execute>

Observation [2]:
The Shanghai Disneyland ticket starts at ￥360.

Assistant [3]:
<thought>
The problem has been solved. I retrieved the hotel price and ticket information using the appropriate APIs, and both results are accurate.
</thought>
Answer: <final_answer>The cheapest Hilton hotel costs ￥557 per night, and the Shanghai Disneyland ticket starts at ￥360.</final_answer>"""
    example_foamat_prompt = """### Example format like this:
Task:
[Task or question]

Assistant [1]:
<thought>
[Reasoning and method to use.]
</thought>
<execute>
```python
[Code or API call for step 1.]
```
</execute>

Observation [1]:
[Result of step 1.]

Assistant [2]:
<thought>
[Reasoning and method for next step.]
</thought>
<execute>
```python
[Code or API call for step 2.]
```
</execute>

Observation [2]:
[Result of step 2.]

...

Assistant [n]:
<thought>
[Summarize reasoning.]
</thought>
Answer: <final_answer>[Final answer.]</final_answer>
"""
    prompt = prompt.replace(example_prompt, example_foamat_prompt)
    return prompt


class IO_Func:
    def __init__(self):
        self.functions = []
        self.tool_description = ""
        self.input_description = ""
        self.code_string = []

    def generate_api_tool(self):

        return self.code_string


backbone_model = "LLM_vllm"
model_name = "qwen2.5_sft1"
SERVICE_URL = "http://localhost:8082/virtual"
strategy = "multi_agent"
vllm_url = "http://0.0.0.0:8003/v1"
save_path_dir = f"/data/user/code/luckytool/data/RefineToolbench/Output/{backbone_model}_{strategy}_{model_name}/"


if not os.path.exists(save_path_dir):
    os.makedirs(save_path_dir)
if backbone_model == "LLM":
    llm_forward = LLM(
        model="gpt3.5",
        openai_key="xxx",
        base_url="baseurl",
    )
elif backbone_model == "LLM_vllm":
    llm_forward = LLM_vllm(model=model_name, openai_key="xxx", base_url=vllm_url)

refine_toolbench_path = "/data/user/code/luckytool/data/RefineToolbench/data"


def process_task(task):
    id, data, save_path = task["id"], task["data"], task["save_path"]
    group = id.split("@")[0]
    io_func = IO_Func()
    io_func.functions = [data["api_doc"]]
    io_func.tool_description = "None"
    io_func.input_description = data["query"]
    data["code_string"] = data["code_string"].replace(
        "http://localhost:8081/virtual", SERVICE_URL
    )
    if group == "I1":
        data["code_string"] = data["code_string"].replace(
            "payload = {", "payload = {'is_api_valid':True,"
        )
    else:
        data["code_string"] = data["code_string"].replace(
            "payload = {", "payload = {'is_api_valid':False,"
        )
    io_func.code_string = [data["code_string"]]

    system = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_WITHOUT_TOOL
    system = system.replace("===api_doc===", str(io_func.functions))
    system = replace_system_prompt(system)
    user = FORMAT_INSTRUCTIONS_USER_FUNCTION
    user = user.replace("===input_description===", io_func.input_description)
    premessages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    wrong_iteration = f"""<thought>
{data["action"]["thought"]}
</thought>
<execute>
```python
{data["action"]["python_code"]}
```
</execute>
"""
    observation = "Observation:\n" + data["action"]["observation"]
    premessages.append({"role": "assistant", "content": wrong_iteration})
    premessages.append({"role": "user", "content": observation})
    chain = multi_agent(llm_forward, io_func, start_message_list=premessages)
    result = chain.start(
        single_chain_max_step=3,
        answer=1,
    )
    with open(save_path, "w") as writer:
        res = chain.to_json(answer=True, process=True)
        res["answer_generation"]["query"] = data["query"]
        json.dump(res, writer, indent=2)


def process_task_I3(task):
    id, data, save_path = task["id"], task["data"], task["save_path"]
    group = id.split("@")[0]
    io_func = IO_Func()
    io_func.functions = data["api_docs"]
    io_func.tool_description = "None"
    io_func.input_description = data["query"]
    data["code_string"] = data["code_string"].replace(
        "http://localhost:8081/virtual", SERVICE_URL
    )
    io_func.code_string = [data["code_string"]]
    premessages = data["pre_messages"]

    wrong_iteration = f"""<thought>
{data["action"]["thought"]}
</thought>
<execute>
```python
{data["action"]["python_code"]}
```
</execute>
"""
    observation = "Observation:\n" + data["action"]["observation"]
    premessages.append({"role": "assistant", "content": wrong_iteration})
    premessages.append({"role": "user", "content": observation})
    chain = multi_agent(llm_forward, io_func, start_message_list=premessages)
    result = chain.start(
        single_chain_max_step=5,
        answer=1,
    )
    with open(save_path, "w") as writer:
        res = chain.to_json(answer=True, process=True)
        res["answer_generation"]["query"] = data["query"]
        json.dump(res, writer, indent=2)


all_task = []
for group in ["I1", "I2"]:
    file_path = os.path.join(refine_toolbench_path, group) + ".json"
    with open(file_path, "r") as f:
        dataset = json.load(f)
    for idx, data in enumerate(dataset):
        save_path = os.path.join(save_path_dir, group + f"@{idx}.json")
        if os.path.exists(save_path):
            continue
        all_task.append({"id": group + f"@{idx}", "data": data, "save_path": save_path})
print(len(all_task))

with Pool(processes=1) as pool:
    pool.map(process_task, all_task)

all_task = []
group = "I3"
file_path = os.path.join(refine_toolbench_path, group) + ".json"
with open(file_path, "r") as f:
    dataset = json.load(f)
for idx, data in enumerate(dataset):
    save_path = os.path.join(save_path_dir, group + f"@{idx}.json")
    if os.path.exists(save_path):
        continue
    all_task.append({"id": group + f"@{idx}", "data": data, "save_path": save_path})
print(len(all_task))
with Pool(processes=1) as pool:
    pool.map(process_task_I3, all_task)
