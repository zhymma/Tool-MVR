import json
import os
from utils.toolbench_utils import standardize, change_name
from utils.utils import chat_completion

# Read the JSON file
import subprocess
from multiprocessing import Pool
from tqdm import tqdm

apigen_tools = json.load(open("data/APIGen/all_tools.json", "r"))


apigen_tools_dict = {}
for tool in apigen_tools:
    name = tool["name"]
    if name in apigen_tools_dict:
        apigen_tools_dict[name].append(tool)
    else:
        apigen_tools_dict[name] = [tool]

# print(change_name(standardize("T3MA")))


def main1():
    # Process each group (G1, G2, G3)
    for group in ["G1", "G2", "G3"]:

        # Read queries
        with open(f"data/instruction/new_{group}_query.json", "r") as f:
            queries = json.load(f)
        no_description_cnt = 0
        # Initialize list to store u    nique tools
        group_tools = []

        # Process each solvable query
        for query in queries:
            # Get tools for this query if they exist
            if "api_list" in query:
                query_tools = query["api_list"]
                for tool in query_tools:
                    if tool not in group_tools:
                        if len(tool["api_description"].strip()) == 0:
                            no_description_cnt += 1
                        group_tools.append(tool)

        print(f"Found {len(group_tools)} unique tools in solvable {group} queries")
        print(f"No description cnt: {no_description_cnt}")
        # Save tools for this group
        with open(f"data/process_data/solvable_api/{group}_tools.json", "w") as f:
            json.dump(group_tools, f, indent=2)


def process_parameters(parameters):
    res = []
    for key, value in parameters.items():
        value["name"] = key
        res.append(value)
    return res


def main2():
    for group in ["G1", "G2", "G3"]:
        with open(f"data/instruction/new_{group}_query.json", "r") as f:
            queries = json.load(f)
        cnt = 0
        all_tool_cnt = 0
        for query in queries:
            group_tools = query["api_list"]
            all_tool_cnt += len(group_tools)
            for tool in group_tools:
                api_name = tool["api_name"]
                standardize_name = change_name(standardize(api_name))
                if standardize_name in apigen_tools_dict:
                    if len(apigen_tools_dict[standardize_name]) > 1:
                        # 通过para判断是哪一个api
                        for api in apigen_tools_dict[standardize_name]:
                            parameters = api["parameters"]
                            parameter_names = list(parameters.keys())
                            tool_parameter_names = [
                                change_name(standardize(para["name"]))
                                for para in tool["required_parameters"]
                                + tool["optional_parameters"]
                            ]
                            if set(parameter_names) == set(tool_parameter_names):
                                tool["api_description"] = api["description"]
                                for para in tool["required_parameters"]:
                                    para["description"] = parameters[
                                        change_name(standardize(para["name"]))
                                    ]["description"]
                                for para in tool["optional_parameters"]:
                                    para["description"] = parameters[
                                        change_name(standardize(para["name"]))
                                    ]["description"]
                                cnt += 1
                                break
                    else:
                        description = apigen_tools_dict[standardize_name][0][
                            "description"
                        ]

                        parameters = apigen_tools_dict[standardize_name][0][
                            "parameters"
                        ]
                        parameter_names = list(parameters.keys())
                        # parameter_names = [change_name(standardize(name))]
                        tool_parameter_names = [
                            change_name(standardize(para["name"]))
                            for para in tool["required_parameters"]
                            + tool["optional_parameters"]
                        ]
                        if set(parameter_names) == set(tool_parameter_names):
                            tool["api_description"] = description
                            for para in tool["required_parameters"]:
                                para["description"] = parameters[
                                    change_name(standardize(para["name"]))
                                ]["description"]
                            for para in tool["optional_parameters"]:
                                para["description"] = parameters[
                                    change_name(standardize(para["name"]))
                                ]["description"]
                            cnt += 1
        print(f"Found {cnt} tools in {group} {all_tool_cnt}")
        # 保存到json
        with open(f"data/instruction/new_{group}_query.json", "w") as f:
            json.dump(queries, f, indent=2)


def main3():  # 测试集也要改一下 api doc
    for group in [
        "G1_instruction",
        "G1_category",
        "G1_tool",
        "G2_category",
        "G2_instruction",
        "G3_instruction",
    ]:
        with open(
            f"stabletoolbench/solvable_queries/test_instruction/{group}.json", "r"
        ) as f:
            queries = json.load(f)
        cnt = 0
        all_tool_cnt = 0
        for query in queries:
            group_tools = query["api_list"]
            all_tool_cnt += len(group_tools)
            for tool in group_tools:
                api_name = tool["api_name"]
                standardize_name = change_name(standardize(api_name))
                if standardize_name in apigen_tools_dict:
                    if len(apigen_tools_dict[standardize_name]) > 1:
                        # 通过para判断是哪一个api
                        for api in apigen_tools_dict[standardize_name]:
                            parameters = api["parameters"]
                            parameter_names = list(parameters.keys())
                            tool_parameter_names = [
                                change_name(standardize(para["name"]))
                                for para in tool["required_parameters"]
                                + tool["optional_parameters"]
                            ]
                            if set(parameter_names) == set(tool_parameter_names):
                                tool["api_description"] = api["description"]
                                for para in tool["required_parameters"]:
                                    para["description"] = parameters[
                                        change_name(standardize(para["name"]))
                                    ]["description"]
                                for para in tool["optional_parameters"]:
                                    para["description"] = parameters[
                                        change_name(standardize(para["name"]))
                                    ]["description"]
                                cnt += 1
                                break
                    else:
                        description = apigen_tools_dict[standardize_name][0][
                            "description"
                        ]

                        parameters = apigen_tools_dict[standardize_name][0][
                            "parameters"
                        ]
                        parameter_names = list(parameters.keys())
                        # parameter_names = [change_name(standardize(name))]
                        tool_parameter_names = [
                            change_name(standardize(para["name"]))
                            for para in tool["required_parameters"]
                            + tool["optional_parameters"]
                        ]
                        if set(parameter_names) == set(tool_parameter_names):
                            tool["api_description"] = description
                            for para in tool["required_parameters"]:
                                para["description"] = parameters[
                                    change_name(standardize(para["name"]))
                                ]["description"]
                            for para in tool["optional_parameters"]:
                                para["description"] = parameters[
                                    change_name(standardize(para["name"]))
                                ]["description"]
                            cnt += 1
        print(f"Found {cnt} tools in {group} {all_tool_cnt}")
        # 保存到json
        with open(
            f"stabletoolbench/solvable_queries/test_instruction/new_{group}.json", "w"
        ) as f:
            json.dump(queries, f, indent=2)


def execute_code(code, temp_file_path="stabletoolbench/code_exec/temp_code_exec.py"):
    max_attempts = 1
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


def main4():
    all_api = {}
    for group in ["G1", "G2", "G3"]:

        data_path = f"data/process_data/all_apis/{group}/"
        for file in os.listdir(data_path):
            with open(os.path.join(data_path, file), "r") as f:
                data = json.load(f)
                for idx, api in enumerate(data["apis"]):
                    name = api["function"]["name"]
                    code_string = data["code_strings"][idx]
                    if name in all_api:
                        continue
                    all_api[name] = {"api": api, "code_string": code_string}
    print(f"Found {len(all_api)} unique tools")
    with open(f"data/process_data/all_apis/all_api.json", "w") as f:
        json.dump(all_api, f, indent=2)


get_example_prompt = """
Please generate a few example function calls based on the following API documentation, using keyword arguments to fully understand the API functionality. The output should be in Python format with printed results in code blocks. Maximum 3 example calls, and results must be printed. Note: No need to import API package names, just call the API functions directly. Note: If the API has no parameters, just call the API function directly without any parameters, otherwise, it will be wrong!!!


## API Call Examples (For API `ticket_info_query` with parameters `departure`,`destination` and `travel_mode`)
```python
print("Example 1:")
print(ticket_info_query(destination="Beijing", travel_mode="Train"))
print("Example 2:")
print(ticket_info_query(departure="Beijing", destination="Shanghai", travel_mode="Plane"))
```

---
Now, for a new API, please generate the API call examples based on the following API documentation.
## API Documentation
# ===api_name===
===api_doc===

## API Call ()

"""


refine_api_doc_prompt_template = """
You are a expert in API documentation. Now we have a few API call results. Please refine the API description based on the API call results to better describe the API functionality. Note you can only change the API description and parameters description, you can't change the API name and parameters name and you can't add or remove parameters.

If all API results show `"is_real_api": false`, it indicates that the API is no longer functional. In such cases, return:
```json
{
    "is_api_valid": false
}
```

If API results include `"is_real_api": true`, but the response has no real data, just return something like sorry/You are not subscribed to this API/The API is not available...
In such cases, you should return:
```json
{
    "is_api_valid": false
}
```
If any API results include `"is_real_api": true`, then based on the actual API request, refine the API description if needed. For instance, update parameter descriptions, or details about the response example, or clarify how to avoid errors and empty returns in description. 


If not need refine the API, return:
```json
{
    "is_api_valid": true
}
```

If need refine the API, return:
```json
{
    "is_api_valid": true,
    "refine_api": {"type": "function",
      "function": {
        "name": "...",
        "description": "...",
        "parameters": {"type": "object", "properties": {"name": "...", "description": "..."}, "required": ["..."], "optional": ["..."]}}
}
```
refine_api is the refined API in OpenAI function calling format. You should make sure the refine_api is correct.

### API Information
===api_name===

### Original API Documentation
===api_doc===

### API Call
===api_call===

### Observation
===Observation===

### Response

"""


refine_api_doc_prompt_template_1 = """
You are a expert in API documentation. Now we have a few API call results. Please check if the API is valid.

If all API results show `"is_real_api": false`, it indicates that the API is no longer functional. In such cases, return:
```json
{
    "is_api_valid": false
}
```

If API results include `"is_real_api": true`, but the response has no real data, just return something like sorry/You are not subscribed to this API/The API is not available...
In such cases, you should return:
```json
{
    "is_api_valid": false
}
```

If the API response real data, return:
```json
{
    "is_api_valid": true
}
```

### API Information
===api_name===

### Original API Documentation
===api_doc===

### API Call
===api_call===

### Observation
===Observation===

### Response
（return a valid json containing is_api_valid, and can be parsed by json.loads）
"""


refine_api_doc_prompt_template_2 = """
You are a expert in API documentation. Now we have a few API call results. Please check if the API is valid.


If the API response is empty or no useful information or always has something like sorry/You are not subscribed to this API/The API is not available...
In such cases, you should return:
```json
{
    "is_api_valid": false
}
```

If the API response has any real information data or API response the api call is wrong because of the invalid parameters passed in or sometimes the API is available and sometimes the API is not available, which means the API is valid and we should use it in a correct way, return:
```json
{
    "is_api_valid": true
}
```

### API Information
===api_name===

### Original API Documentation
===api_doc===

### API Call
===api_call===

### Observation
===Observation===

### Response
（return a valid json containing is_api_valid, and can be parsed by json.loads）
"""


import random


def test_api(name, api):
    # 测试api的调用
    # print(api['api']['function']['description'])
    # print(api['code_string'])
    try:
        if os.path.exists(f"data/process_data/all_apis/backup/{name}.json"):
            with open(f"data/process_data/all_apis/backup/{name}.json", "r") as f:
                api = json.load(f)
            return name, api
        code_string = api["code_string"]

        prompt = get_example_prompt.replace("===api_name===", name).replace(
            "===api_doc===", str(api["api"])
        )
        models = ["gpt-4-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-11-20"]
        model = random.choice(models)
        res = chat_completion(
            "xxx",
            [{"role": "user", "content": prompt}],
            "baseurl",
            model=model,
        )
        # print(res)
        if "error" in res:
            print(f"Error: {res['error']}")
            return None
        api_call = res["choices"][0]["message"]["content"]
        if "```python" in api_call:
            api_call = api_call.split("```python")[1].split("```")[0]
        print(api_call)
        if "timeout=timeout" in code_string:
            code_string = code_string.replace("timeout=timeout", "timeout=15")
        code = code_string + "\n" + api_call

        result = execute_code(code)
        print(result, "\n\n")
        if "Note: An error occurred" in result:
            print(f"Error: {result}")
            return None
        api["api_call"] = api_call
        api["call_result"] = result

        models = ["gpt-4-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-11-20"]
        model = random.choice(models)
        refine_api_doc_prompt = (
            refine_api_doc_prompt_template.replace("===api_name===", name)
            .replace("===api_doc===", str(api["api"]))
            .replace("===api_call===", api_call)
            .replace("===Observation===", result)
        )
        res = chat_completion(
            "xxx",
            [{"role": "user", "content": refine_api_doc_prompt}],
            "baseurl",
            model=model,
            temperature=0,
        )
        # print(res)
        res = res["choices"][0]["message"]["content"]

        if "```json" in res:
            res = res.split("```json")[1].split("```")[0]
        res = json.loads(res)

        api["is_api_valid"] = res["is_api_valid"]
        if "refine_api" in res:
            api["refine_api"] = res["refine_api"]
        with open(f"data/process_data/all_apis/backup/{name}.json", "w") as f:
            json.dump(api, f, indent=2)
        return name, api
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_api_1(name, api):
    # 测试api的调用
    # print(api['api']['function']['description'])
    # print(api['code_string'])
    try:
        models = ["gpt-4-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-11-20"]
        model = random.choice(models)
        refine_api_doc_prompt = (
            refine_api_doc_prompt_template_1.replace("===api_name===", name)
            .replace("===api_doc===", str(api["api"]))
            .replace("===api_call===", api["api_call"])
            .replace("===Observation===", api["call_result"])
        )
        res = chat_completion(
            "xxx",
            [{"role": "user", "content": refine_api_doc_prompt}],
            "baseurl",
            model=model,
            temperature=0,
        )
        # print(res)
        res = res["choices"][0]["message"]["content"]

        if "```json" in res:
            res = res.split("```json")[1].split("```")[0]
        res = json.loads(res)

        is_api_valid = res["is_api_valid"]
        if is_api_valid == False:
            print(f"is_api_valid is False: {name}")
        return name, is_api_valid
    except Exception as e:
        print(f"Error: {e}")
        return None


def test_api_2(name, api):
    # 测试api的调用
    # print(api['api']['function']['description'])
    # print(api['code_string'])
    try:
        models = ["gpt-4-turbo", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-11-20"]
        model = random.choice(models)
        refine_api_doc_prompt = (
            refine_api_doc_prompt_template_2.replace("===api_name===", name)
            .replace("===api_doc===", str(api["api"]))
            .replace("===api_call===", api["api_call"])
            .replace("===Observation===", api["call_result"])
        )
        res = chat_completion(
            "xxx",
            [{"role": "user", "content": refine_api_doc_prompt}],
            "baseurl",
            model=model,
            temperature=0,
        )
        # print(res)
        res = res["choices"][0]["message"]["content"]

        if "```json" in res:
            res = res.split("```json")[1].split("```")[0]
        res = json.loads(res)

        is_api_valid = res["is_api_valid"]
        if is_api_valid == True:
            print(f"is_api_valid is True: {name}")
        return name, is_api_valid
    except Exception as e:
        print(f"Error: {e}")
        return None


def main5():
    with open(f"data/process_data/all_apis/all_api.json", "r") as f:
        all_api = json.load(f)

    # Create thread pool with 5 workers
    with Pool(processes=20) as pool:
        # Create list of (name, api) tuples for all APIs
        api_tasks = [(name, api) for name, api in all_api.items()]
        import random

        random.shuffle(api_tasks)
        print(len(api_tasks))
        # api_tasks = api_tasks[:5]
        # Map test_api function across all tasks in parallel
        results = pool.starmap(test_api, api_tasks)

        # Process results and update all_api dict
        for result in results:
            if result is not None:
                name, api_result = result
                all_api[name] = api_result

    # Save updated all_api to file
    with open(f"data/process_data/all_apis/all_api_processed.json", "w") as f:
        json.dump(all_api, f, indent=2)


def main5_1():
    with open(f"data/process_data/all_apis/all_api_processed.json", "r") as f:
        all_api = json.load(f)

    # Create thread pool with 5 workers

    # Create list of (name, api) tuples for all APIs
    api_tasks = [
        (name, api)
        for name, api in all_api.items()
        if "is_api_valid" in api and api["is_api_valid"] == True
    ]
    # api_tasks = api_tasks[:50]
    import random

    random.shuffle(api_tasks)
    print(len(api_tasks))
    total_tasks = len(api_tasks)
    # api_tasks = api_tasks[:5]
    # Map test_api function across all tasks in parallel
    with Pool(processes=30) as pool:
        # Use tqdm to wrap the iterator
        results = list(
            tqdm(
                pool.starmap(test_api_1, api_tasks),
                total=total_tasks,
                desc="Processing APIs",
            )
        )

    fix_cnt = 0
    # Process results and update all_api dict
    for result in results:
        if result is not None:
            name, is_api_valid = result
            if is_api_valid == False:
                all_api[name]["is_api_valid"] = is_api_valid
                fix_cnt += 1
    print(f"Fix cnt: {fix_cnt}")
    # Save updated all_api to file
    with open(f"data/process_data/all_apis/all_api_processed_1.json", "w") as f:
        json.dump(all_api, f, indent=2)


def main5_2():
    with open(f"data/process_data/all_apis/all_api_processed_1.json", "r") as f:
        all_api = json.load(f)

    # Create thread pool with 5 workers

    # Create list of (name, api) tuples for all APIs
    api_tasks = [
        (name, api)
        for name, api in all_api.items()
        if "is_api_valid" in api
        and api["is_api_valid"] == False
        and "refine_api" in api
    ]
    # api_tasks = api_tasks[:50]
    import random

    random.shuffle(api_tasks)
    print(len(api_tasks))
    total_tasks = len(api_tasks)
    # api_tasks = api_tasks[:5]
    # Map test_api function across all tasks in parallel
    with Pool(processes=30) as pool:
        # Use tqdm to wrap the iterator
        results = list(
            tqdm(
                pool.starmap(test_api_2, api_tasks),
                total=total_tasks,
                desc="Processing APIs",
            )
        )

    fix_cnt = 0
    # Process results and update all_api dict
    for result in results:
        if result is not None:
            name, is_api_valid = result
            if is_api_valid == True:
                all_api[name]["is_api_valid"] = is_api_valid
                fix_cnt += 1
    print(f"Fix cnt: {fix_cnt}")
    # Save updated all_api to file
    with open(f"data/process_data/all_apis/all_api_processed_2.json", "w") as f:
        json.dump(all_api, f, indent=2)


def main6():
    final_all_api = {}
    with open(f"data/process_data/all_apis/all_api_processed_2.json", "r") as f:
        all_api = json.load(f)

    refine_cnt = 0
    valid_cnt = 0
    for name, api in all_api.items():
        try:
            if api["is_api_valid"] == False:
                final_all_api[name] = {
                    "api": api["api"],
                    "is_api_valid": False,
                }
            else:
                valid_cnt += 1
                if "refine_api" in api:

                    refine_api = api["refine_api"]
                    original_api = api["api"]
                    is_correct = True
                    if (
                        original_api["function"]["name"]
                        != refine_api["function"]["name"]
                    ):
                        is_correct = False
                    original_p_names = list(
                        original_api["function"]["parameters"]["properties"].keys()
                    )
                    refine_p_names = list(
                        refine_api["function"]["parameters"]["properties"].keys()
                    )
                    if set(original_p_names) != set(refine_p_names):
                        is_correct = False
                    if is_correct:
                        refine_cnt += 1
                        final_all_api[name] = {
                            "api": api["refine_api"],
                            "is_api_valid": True,
                        }
                    else:
                        final_all_api[name] = {
                            "api": api["api"],
                            "is_api_valid": True,
                        }
                else:
                    final_all_api[name] = {
                        "api": api["api"],
                        "is_api_valid": api["is_api_valid"],
                    }
        except Exception as e:
            print(f"Error: {e}")
            print(name)
            final_all_api[name] = {
                "api": api["api"],
                "is_api_valid": False,
            }

    print(
        f"Refine cnt: {refine_cnt}",
        f"Total cnt: {len(all_api)}",
        f"Valid cnt: {valid_cnt}",
    )
    with open(f"data/process_data/all_apis/final_all_api.json", "w") as f:
        json.dump(final_all_api, f, indent=2)




# 1. 统计出所有的tool,function_code,和description
if __name__ == "__main__":
    # main1()
    # main2()
    # main3()
    # main4()

    # main5()
    # main5_1()
    # main5_2()
    main6()
