import json
import os
import random
from utils.utils import *
from multiprocessing import Pool
import tqdm

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
        with open(
            f"stabletoolbench/solvable_queries/test_instruction/G1_category.json", "r"
        ) as f:
            test_queries1 = json.load(f)
        with open(
            f"stabletoolbench/solvable_queries/test_instruction/G1_instruction.json",
            "r",
        ) as f:
            test_queries2 = json.load(f)
        with open(
            f"stabletoolbench/solvable_queries/test_instruction/G1_tool.json", "r"
        ) as f:
            test_queries3 = json.load(f)
        queries = test_queries1 + test_queries2 + test_queries3
    elif group == "G2":
        with open(
            f"stabletoolbench/solvable_queries/test_instruction/G2_category.json", "r"
        ) as f:
            test_queries1 = json.load(f)
        with open(
            f"stabletoolbench/solvable_queries/test_instruction/G2_instruction.json",
            "r",
        ) as f:
            test_queries2 = json.load(f)
        queries = test_queries1 + test_queries2
    elif group == "G3":
        with open(
            f"stabletoolbench/solvable_queries/test_instruction/G3_instruction.json",
            "r",
        ) as f:
            test_queries = json.load(f)
        queries = test_queries
    test_ids = [str(query["query_id"]) for query in queries]
    return test_ids


data_quality_prompt = """
You are an SFT (Supervised Fine-Tuning) data quality engineer.

I will provide you with self-correcting instruction data, where the last two rounds follow the pattern: wrong action -> observation -> right action. The key focus is on generating reflection thoughts from the observation and making corrections.

Please evaluate the quality of this data sample on a scale of 1-10, considering the following aspects. Our goal is to build high-quality self-correcting data that helps models learn to reflect on observations and make appropriate corrections.

Evaluation Criteria:
0. Contextual Semantic Coherence
   - Evaluate whether the previous messages and the last two rounds of messages are coherent and logically consistent
   - Assess if the error occurrence is plausible in the given context
   - Inconsistencies can affect the model's learning distribution space

1. Error Complexity
   - The error should not be too trivial that models can solve without training
   - Higher scores for more sophisticated, training-worthy errors

2. Error Reasonableness
   - Based on earlier iterations, is this the type of error the model would realistically make?
   - Higher scores if the error pattern matches expected model behavior

3. Observation Quality
   - How semantically rich and informative is the observation?
   - Higher scores for detailed, meaningful observations that provide clear correction signals

4. Reflection Quality
   - How thorough and logical is the reflection thought?
   - Does it align well with the subsequent right action?
   - Higher scores for comprehensive reflections with clear reasoning

The types of errors may include:
- **API Selection errors** 
    - Use wrong API name, which different from the correct API name. (More Important):
    - Use wrong API parameter value, which different from the correct API parameter value, and will lead to wrong result. (More Important):
    - Usage of non-existent APIs, which not in the provided API list. (Easy to detect)
- **API parameter errors**:
    - Missing required parameters. (Easy to detect)
    - Invalid input parameters.
    - Parameter type errors such as using string for integer parameter or wrong date format. (Hard to detect)
    - Parameter value exceeding the valid range. (Hard to detect)
    - Correctly formatted parameters but logically meaningless for the query. (More important)

Note that you must be sufficiently strict, as this is crucial for my training. Your scores should aim for an even distribution across the 1-10 range, with roughly equal numbers of low (1-3), medium (4-7), and high (8-10) scores based on the evaluation criteria.
---

# Dataset
===messages===

# Response Format: (Output format as json containing score and reason, in ```json format)
```json
{
    "score": 1-10,
    "reason": "reason"
}
```
"""
all_task = []
for group in ["G1", "G2", "G3"]:
    test_ids = get_test_ids(group)
    folder_path = os.path.join(base_path, group)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if (
                            data.get("win") == True
                            and (
                                data.get("answer_status").get("answer_status") == "Pass"
                                or data.get("answer_status").get("answer_status")
                                == "Solved"
                            )
                            and (
                                data.get("answer_status").get("all_steps_validity")
                                == "yes"
                                or data.get("answer_status").get("all_steps_validity")
                                == True
                                or data.get("answer_status").get("all_steps_validity")
                                == "True"
                                or data.get("answer_status").get("all_steps_validity")
                                == "true"
                            )
                        ):
                            id = filename.split(".")[0]
                            id = id.split("_")[0]
                            if id in test_ids:
                                continue
                            messages = data["answer_generation"]["messages"]
                            wrong_api_call_index = data["wrong_iteration"][
                                "wrong_api_call_index"
                            ]
                            pre_messages = messages[:wrong_api_call_index]

                            wrong_action = (
                                "<thought>\n"
                                + data["wrong_iteration"]["action"]["thought"]
                                + "\n</thought>\n<execute>\n"
                                + data["wrong_iteration"]["action"]["execute"]
                                + "\n</execute>"
                            )
                            wrong_message = [
                                {"role": "assistant", "content": wrong_action}
                            ]
                            observation = data["wrong_iteration"]["observation"]
                            if (
                                "Observation:\nError: Failed to run the code after 3 attempts.\nError: Traceback ("
                                in observation
                            ):
                                continue
                            if observation == "":
                                continue
                            if "{'error': '', 'response': '{}'}" in observation:
                                continue
                            if (
                                "print"
                                not in data["wrong_iteration"]["action"]["execute"]
                            ):
                                continue
                            if "reflextion_data_quality" in data:
                                continue

                            pre_messages = (
                                pre_messages
                                + wrong_message
                                + [
                                    {
                                        "role": "user",
                                        "content": "Observation:\n" + observation,
                                    }
                                ]
                            )
                            right_action = (
                                "<thought>\n"
                                + data["wrong_iteration"]["reflection_iteration"][
                                    "action"
                                ]["thought"]
                                + "\n</thought>\n<execute>\n"
                                + data["wrong_iteration"]["reflection_iteration"][
                                    "action"
                                ]["execute"]
                                + "\n</execute>"
                            )

                            all_messages = pre_messages + [
                                {
                                    "role": "assistant",
                                    "content": right_action,
                                    "no_loss": False,
                                }
                            ]
                            all_messages = all_messages[1:]

                            all_task.append(
                                {
                                    "id": str(group) + "@" + id + "_reflection",
                                    "messages": all_messages,
                                    "file_path": file_path,
                                }
                            )
                except:
                    continue


print(len(all_task))


def process_task(task):
    try:
        messages = task["messages"]
        prompt = data_quality_prompt.replace("===messages===", json.dumps(messages))
        models = ["gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-11-20"]
        model = random.choice(models)
        result = chat_completion(
            "xxx",
            [{"role": "user", "content": prompt}],
            "baseurl",
            model=model,
            temperature=0.0,
        )
        content = result["choices"][0]["message"]["content"]
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        else:
            content = content.strip("{}")
            content = "{" + content + "}"
        result = json.loads(content)

        with open(task["file_path"], "r") as f:
            data = json.load(f)
        data["reflextion_data_quality"] = result
        with open(task["file_path"], "w") as f:
            json.dump(data, f, indent=4)
        print(task["id"], result["score"], result["reason"], task["file_path"])
        return 1
    except:
        return 0


# Create a process pool with 8 workers
with Pool(50) as pool:
    # Process tasks in parallel with progress bar
    list(tqdm.tqdm(pool.imap_unordered(process_task, all_task), total=len(all_task)))

294903
