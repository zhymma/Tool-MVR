import json
import random
import os
from utils.utils import chat_completion
import subprocess
import os

import json


def execute_code(code, temp_file_path="stabletoolbench/code_exec/temp_code_exec.py"):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Write the code to a temporary file
            with open(temp_file_path, "w", encoding="utf-8") as file:
                file.write(code)

            # Execute the code using subprocess
            result = subprocess.run(
                ["python", temp_file_path], capture_output=True, text=True, timeout=60
            )

            # Check if there was an error
            if result.returncode != 0:
                raise Exception(
                    f"Error: {result.stderr}\n\nNote: An error occurred during execution. The API call parameters may need to be adjusted, or try calling a different API."
                )
            # if "'errCode': 0, 'errMsg': 'succ', 'data': {}" in result.stdout:
            #     raise Exception(
            #         f"Error: {result.stderr}\n\nNote: The API call returned empty results. The parameters may need to be adjusted according to the API documentation, or try calling a different API."
            #     )
            # Return the output
            output = result.stdout.strip()
            if len(output) > 8192:
                output = output[:8192] + "..."
            return output
        except Exception as e:
            if attempt == max_attempts - 1:
                return (
                    "Error: Failed to run the code after 3 attempts.\n"
                    + str(e)
                    + "\n\nNote: An error occurred or empty results were returned. The API call parameters may need to be adjusted, or try calling a different API."
                )


data_path = "stabletoolbench/data_eval/answer/new_train_0106"
done_num = 0
all_tasks = []
for group in ["G3", "G2", "G1"]:
    folder_path = os.path.join(data_path, group)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if (
                        data.get("win") == True
                        and "answer_status" in data
                        and data.get("answer_status").get("answer_status") == "Pass"
                        and (
                            data.get("answer_status").get("all_steps_validity") == "yes"
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
                        final_answer = data["answer_generation"]["final_answer"]
                        query = data["answer_generation"]["query"]
                        messages = data["answer_generation"]["messages"]
                        action_indexs = [
                            i
                            for i, message in enumerate(messages)
                            if message.get("role") == "assistant"
                            and "<execute>" in message.get("content")
                            and "</execute>" in message.get("content")
                        ]
                        if len(action_indexs) == 0:
                            continue
                        if "functions_strings" not in data:
                            continue
                        if "reflection_iteration" in data["wrong_iteration"]:
                            continue
                        wrong_api_call_index = random.choice(action_indexs)
                        all_tasks.append(
                            {
                                "id": str(group) + "@" + id,
                                "query": query,
                                "wrong_api_call_index": wrong_api_call_index,
                                "final_answer": final_answer,
                                "file_path": file_path,
                            }
                        )
print(len(all_tasks))
print(done_num)

# exit()
# 从训练集的成功的轨迹中第二条到第五条之间截断，并提示LLM下一步生成错误的API调用（用于构造RefineToolbench)
# 应该生成错误的thought action，同时生成一个reflextion和正确的action
# 还得得到错误的observation呢，看来这里也需要真实的探索，同时需要execute(code)和code_string

get_wrong_api_calling_prompt = """
As you can see, I have provided you with a query and several initial iterations (which may be empty). Now I need you to generate an incorrect next iteration to construct a test case for verifying llm's self-correction ability. I can provide you with the correct next iteration as a reference.

### Instructions:
1. **Understand the query and API's docs in system prompt and several iterations**:
   - Api docs in system prompt has many APIs with docs.
   - Several iterations are the previous iterations of solving the query.

2. **Generate Incorrect Iteration in Format**:
   You must introduce an incorrect API calling in the iteration.
   <thought>
   Explain reasoning for the API call, even though it will be incorrect but looks correct.
   </thought>
   <execute>
   ```python
   print(get_flight_prices(origin="Los Angeles", destination="Tokyo", date="2025-02-10"))
   ```
   </execute>
   The error type should be diverse, such as
    - **API Selection errors**:
       - Use wrong API name, which different from the correct API name.
       - Use wrong API parameter value, which different from the correct API parameter value, and will lead to wrong result.
       - Usage of non-existent APIs, which not in the provided API list.
    - **API parameter errors**:
       - Missing required parameters.
       - Invalid input parameters.
       - Parameter type errors such as using string for integer parameter or wrong date format.
       - Parameter value exceeding the valid range.
       - Correctly formatted parameters but logically meaningless for the query.
    
    The Execute code should be one line of code in python (without any comments).
---

### Core Rules:
- **must** include:
  - <thought> It should be a sentence looks right, do not include any hints of the wrong action.
  - <execute> It must has wrong API call.
  
You should simulate a very weak LLM, which cannot choose the right API and understand the API parameters, to simulate its wrong cases in diverse ways.

### Notes:
1. Ensure that at least one incorrect API call is generated during reasoning.
2. The query should be diverse and the wrong API call should be hard to be corrected by LLM, because we want to test the ability of LLM to correct the incorrect API call !!!!!!
---

# Correct iteration
===correct_iteration===

# Wrong iteration
[think step by step and generate the wrong iteration.]
<thought>
xxx
</thought>
<execute>
```python
xxx
```
</execute>
"""

get_reflection_prompt = """
As you can see, I have provided you with a query and several initial iterations, the last iteration is the wrong iteration. Now I need you to generate a reflection for the wrong iteration.

### Instructions:
1. **Understand the query and API's docs in system prompt and several iterations**:
   - Api docs in system prompt has many APIs with docs.
   - Several iterations are the previous iterations of solving the query.

2. **Understand Incorrect Iteration (wrong iteration)**:
   You must understand the wrong iteration with (<thought> and <execute>). And the last observation is the feedback from the wrong iteration.
   The error type may be diverse, such as
    - **API Selection errors**:
       - Use wrong API name, which different from the correct API name.
       - Use wrong API parameter value, which different from the correct API parameter value, and will lead to wrong result.
       - Usage of non-existent APIs, which not in the provided API list.
    - **API parameter errors**:
       - Missing required parameters.
       - Invalid input parameters.
       - Parameter type errors such as using string for integer parameter or wrong date format.
       - Parameter value exceeding the valid range.
       - Correctly formatted parameters but logically meaningless for the query.
3. **Generate the Reflection iteration in Format (which will be used to train the LLM for self-correction with the observation feedback)**:
   You must generate a reflection for the wrong iteration.
   <thought>
   From the observation feedback, analyze the wrong action and the reason why it is wrong (the observation may be execution error or execution result, if it is execution error, it may be the wrong API call or the wrong API parameter value, if it is execution result, it may be meaningless API call and do not contribute to solve the query). Then think step by step to plan now subtask to generate the correct action.
   </thought>
   <execute>
   ```python
   print(xxx)
   ```
   </execute>
---

### Core Rules:
- **must** include:
  - <thought> 
  - <execute>

You should simulate a very self-correcting LLM, which can choose the right API and understand the API parameters, to reflect the wrong action and generate the correct action.

### Notes:
1. The reflection thought should be should be thorough and meticulous, which will be used to train the LLM for self-correction with the observation feedback.
2. The reflection thought should be based on the observation feedback to actively discover errors, rather than knowing in advance that the previous iteration was wrong. The observation feedback may include execution errors or execution results (which are meaningless for solving the problem, indicating that the wrong API was chosen in the previous round). In summary, the reflection thought should be based on observation feedback to actively discover errors, rather than knowing beforehand that the previous iteration was wrong.

---

Also , I provide you with the right thought and action you can use as a reference.
===right iteration===

---

Now, It's your turn to generate the reflection for the self-correcting iteration!!!
# Reflection
<thought>
xxx
</thought>
<execute>
```python
xxx
```
</execute>
"""


def process_task(task):
    try:
        id = task["id"]
        file_path = task["file_path"]
        with open(file_path, "r") as f:
            data = json.load(f)
        wrong_api_call_index = task["wrong_api_call_index"]
        messages = data["answer_generation"]["messages"]

        right_action = messages[wrong_api_call_index]["content"]

        # todo
        prompt = get_wrong_api_calling_prompt.replace(
            "===correct_iteration===", str(right_action)
        )

        new_messages = messages[:wrong_api_call_index] + [
            {"role": "user", "content": prompt}
        ]

        models = ["gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-11-20"]
        model = random.choice(models)
        response = chat_completion(
            key="xxx",
            messages=new_messages,
            base_url="baseurl",
            temperature=0.7,
            model=model,
        )

        content = response["choices"][0]["message"]["content"]

        # Extract thought and execute from incorrect API call
        thought_start = content.find("<thought>") + len("<thought>")
        thought_end = content.find("</thought>")
        thought = content[thought_start:thought_end].strip()

        execute_start = content.find("<execute>") + len("<execute>")
        execute_end = content.find("</execute>")
        execute = content[execute_start:execute_end].strip()

        result = {
            "wrong_api_call_index": wrong_api_call_index,
            "content": content,
            "action": {"thought": thought, "execute": execute},
            "right_action": right_action,
        }
        data["wrong_iteration"] = result
        # Save result to JSON file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        print(id, wrong_api_call_index, content)
        print("saved to ", file_path)
        return result

    except Exception as e:
        print(e)
        return None


def process_task_get_observation(task):
    try:
        id = task["id"]
        file_path = task["file_path"]
        with open(file_path, "r") as f:
            data = json.load(f)
        functions_strings = data["functions_strings"]
        functions_strings = functions_strings.replace(
            "http://localhost:8081/virtual", "http://localhost:8082/virtual"
        )
        wrong_action = data["wrong_iteration"]["action"]["execute"]
        if "```python" in wrong_action:
            wrong_action = wrong_action.replace("```python", "").replace("```", "")
        code_string = functions_strings + "\n" + wrong_action
        result = execute_code(code_string)
        data["wrong_iteration"]["observation"] = result
        # Save result to JSON file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        print(id, result)
        print("saved to ", file_path)
        return result

    except Exception as e:
        print(e)
        print(file_path)
        return None


def process_task_get_reflection(task):
    try:
        id = task["id"]
        file_path = task["file_path"]
        with open(file_path, "r") as f:
            data = json.load(f)
        wrong_api_call_index = data["wrong_iteration"]["wrong_api_call_index"]
        messages = data["answer_generation"]["messages"]
        right_action = data["wrong_iteration"]["right_action"]
        wrong_action = (
            "<thought>\n"
            + data["wrong_iteration"]["action"]["thought"]
            + "\n</thought>\n<execute>\n"
            + data["wrong_iteration"]["action"]["execute"]
            + "\n</execute>"
        )
        observation = data["wrong_iteration"]["observation"]
        prompt = get_reflection_prompt.replace(
            "===right_iteration===", str(right_action)
        )
        messages = (
            messages[:wrong_api_call_index]
            + [{"role": "assistant", "content": wrong_action}]
            + [{"role": "user", "content": "Observation:\n" + observation}]
            + [{"role": "user", "content": prompt}]
        )
        [{"role": "user", "content": prompt}]
        models = ["gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-11-20"]
        model = random.choice(models)
        response = chat_completion(
            key="xxx",
            messages=messages,
            base_url="baseurl",
            temperature=0.7,
            model=model,
        )
        content = response["choices"][0]["message"]["content"]
        thought_start = content.find("<thought>") + len("<thought>")
        thought_end = content.find("</thought>")
        thought = content[thought_start:thought_end].strip()

        execute_start = content.find("<execute>") + len("<execute>")
        execute_end = content.find("</execute>")
        execute = content[execute_start:execute_end].strip()
        reflection_iteration = {
            "content": content,
            "action": {"thought": thought, "execute": execute},
        }
        data["wrong_iteration"]["reflection_iteration"] = reflection_iteration
        # Save result to JSON file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        print(id, content)
        print("saved to ", file_path)
        return reflection_iteration

    except Exception as e:
        print(e)
        print(file_path)
        return None


from multiprocessing import Pool, Manager


if __name__ == "__main__":
    manager = Manager()
    correct_num = manager.Value("i", 0)

    with Pool(processes=10) as pool:
        results = pool.map(process_task_get_reflection, all_tasks)
        for result in results:
            if result is not None:
                correct_num.value += 1

    print(correct_num.value)
