import json
import os

base_path = "stabletoolbench/data_eval/answer/new_train_0106"

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

def get_test_ids(group):
    # Load original queries
    if group == "G1":
        with open(f"stabletoolbench/solvable_queries/test_instruction/G1_category.json", "r") as f:
            test_queries1 = json.load(f)
        with open(f"stabletoolbench/solvable_queries/test_instruction/G1_instruction.json", "r") as f:
            test_queries2 = json.load(f)
        with open(f"stabletoolbench/solvable_queries/test_instruction/G1_tool.json", "r") as f:
            test_queries3 = json.load(f)
        queries = test_queries1 + test_queries2 + test_queries3
    elif group == "G2":
        with open(f"stabletoolbench/solvable_queries/test_instruction/G2_category.json", "r") as f:
            test_queries1 = json.load(f)
        with open(f"stabletoolbench/solvable_queries/test_instruction/G2_instruction.json", "r") as f:
            test_queries2 = json.load(f)
        queries = test_queries1 + test_queries2
    elif group == "G3":
        with open(f"stabletoolbench/solvable_queries/test_instruction/G3_instruction.json", "r") as f:
            test_queries = json.load(f)
        queries = test_queries
    test_ids = [str(query['query_id']) for query in queries]
    return test_ids

train_data_message = {"G1":[], "G2":[], "G3":[]}
train_data = []
for group in ["G1", "G2", "G3"]:
    test_ids = get_test_ids(group)
    folder_path = os.path.join(base_path, group)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if data.get('win') == True and (data.get('answer_status').get('answer_status') == "Pass" or data.get('answer_status').get('answer_status') == "Solved") and (data.get('answer_status').get('all_steps_validity') == "yes" or data.get('answer_status').get('all_steps_validity') == True or data.get('answer_status').get('all_steps_validity') == "True" or data.get('answer_status').get('all_steps_validity') == "true"):
                            id = filename.split(".")[0]
                            id = id.split("_")[0]
                            if id in test_ids:
                                continue
                            train_data_message[group].append(id)
                            messages = data["answer_generation"]["messages"]
                            messages[0]["content"] = replace_system_prompt(messages[0]["content"])
                            train_data.append({"id": str(group)+"@"+id, "messages": messages})
                except:
                    continue

# 输出统计信息
win_count = 0
total_count = 0
for group in ["G1", "G2", "G3"]:
    win_count += len(train_data_message[group])
    total_count += len(os.listdir(os.path.join(base_path, group)))
    print(f"{group}: Win count: {len(train_data_message[group])}, Total count: {len(os.listdir(os.path.join(base_path, group)))}")
print(f"Total: Win count: {win_count}, Total count: {total_count}")
from datetime import datetime

current_time = datetime.now().strftime("%m%d_%H")
with open(f"data/training_data/0108_10_training_data.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open(f"data/training_data/0108_10_data_info.json", "w") as f:
    json.dump(train_data_message, f, indent=4)


