import json 
import random
from utils.utils import chat_completion
import json 
import random
import os
from utils.utils import chat_completion
import subprocess
import os
from tqdm import tqdm
import ast



with open("data/process_data/all_apis/final_all_api.json", "r") as f:
    all_apis = json.load(f)
with open("data/process_data/all_apis/all_api.json", "r") as f:
    all_apis_info = json.load(f)
# Get all files in APICallingSaves directory
api_calling_dir = "data/RefineToolbench/APICallingSaves"
valid_apis = []
invalid_apis = []

# 写一个函数，将```python\nprint(manga_scrape(slug=\"the-world-after-the-fall-chapter-64\", provider=\"flame\", webtoon=\"the-world-after-the-fall\"))\n```转化为标准的json function calling
def convert_to_json_function_calling(code):
    # 去掉```python\n和\n```
    if "```python" in code:
        code = code.replace("```python\n", "").replace("\n```", "").strip()
    # 去掉print(和最后一个)
    if "print(" in code:
        code = code.replace("print(", "")
        # Find last closing parenthesis
        last_paren_idx = code.rindex(')')
        code = code[:last_paren_idx]
    args_dict = {}
    function_name= ""
    try:
        # 使用 ast 解析代码
        parsed = ast.parse(code, mode='eval')
        call_node = parsed.body
        
        # 获取函数名
        function_name = call_node.func.id
        
        # 收集关键字参数
        

        for keyword in call_node.keywords:
            key = keyword.arg
            value = ast.literal_eval(keyword.value)
            args_dict[key] = value
    except Exception as e:
        # 解析失败时返回空结果
        print(f"Error: {e}")

    if args_dict == {}:
        # 尝试使用一般方法
        
        function_name = code.split("(")[0]
        args_str = code[code.find("(")+1:code.rindex(")")] 
        args_dict = {}
        for arg in args_str.split(","):
            arg = arg.strip()
            if "=" in arg:
                    key, value = arg.split("=")
                    args_dict[key] = value

    return {
        "api_name": function_name,
        "parameters": args_dict
    }
    



for idx, filename in enumerate(os.listdir(api_calling_dir)):
    if filename.endswith('.json'):
        with open(os.path.join(api_calling_dir, filename), "r") as f:
            data = json.load(f)
        api_name = filename.split(".")[0]
        if api_name == "facilities_lookup_for_tunisia_api":
            continue
        if data["incorrect_api_call"]["observation"] == "":
            continue
        if data["incorrect_api_call"]["observation"] == "{'error': '', 'response': '{}'}":
            continue
        python_code = data["incorrect_api_call"]["execute"]
        if "none" in python_code:
            python_code = python_code.replace("none", "None")
        process_item = {}
        query=data["query"]
        if "<query>" in query:
            query = query.replace("<query>", "").replace("</query>", "")
        process_item["query"] = query
        action = {
            "thought": data["incorrect_api_call"]["thought"],
            "python_code": python_code.replace("```python", "").replace("```", ""),
            "json_function_calling": convert_to_json_function_calling(python_code),
            "observation": data["incorrect_api_call"]["observation"]
        }
        if len(action["json_function_calling"]["parameters"]) == 0:
            continue
        if action["json_function_calling"]["api_name"] == "":
            continue
        process_item["action"] = action
        process_item["answer"] = data["correct_api_call"]
        process_item["api_name"] = api_name
        process_item["api_doc"] = all_apis[api_name]["api"]
        process_item["code_string"] = all_apis_info[api_name]["code_string"]
        process_item["data_path"] = os.path.join(api_calling_dir, filename)
        # Check if API exists in all_apis
        if api_name in all_apis:
            # Check if API is valid according to all_apis
            if all_apis[api_name].get("is_api_valid", False):
                valid_apis.append(process_item)
            else:
                invalid_apis.append(process_item)

print(f"Found {len(valid_apis)} valid APIs and {len(invalid_apis)} invalid APIs")

with open("data/RefineToolbench/data/I1.json", "w") as f:
    json.dump(valid_apis, f, indent=4)
with open("data/RefineToolbench/data/I2.json", "w") as f:
    json.dump(invalid_apis, f, indent=4)

cnt = 0


api_trace_list = []
api_trace_dir = "data/RefineToolbench/APITraceSaves"
for idx, filename in enumerate(os.listdir(api_trace_dir)):
    if filename.endswith('.json'):
        with open(os.path.join(api_trace_dir, filename), "r") as f:
            data = json.load(f)
        try:
            query = data["answer_generation"]["query"]
            if "<query>" in query:
                query = query.replace("<query>", "").replace("</query>", "")
            process_item = {}
            process_item["query"] = query
            process_item["api_docs"] = data["answer_generation"]["function"]
            if "functions_strings" in data:
                process_item["code_string"] = data["functions_strings"]
            else:
                cnt += 1
                continue
            wrong_idx = data["wrong_iteration"]["wrong_api_call_index"]
            pre_messages = data["answer_generation"]["messages"][:wrong_idx]
            wrong_action_id = (wrong_idx - 2) // 2
            pre_actions = data["answer_generation"]["actions"][:wrong_action_id]
            new_pre_actions = []
            is_invalid_data = True
            for action in pre_actions:
                if action["type"] == "execute":
                    temp = {}
                    temp["thought"] = action["thought"]
                    temp["python_code"] = action["code"].strip()
                    temp["json_function_calling"] = convert_to_json_function_calling(action["code"].strip())
                    temp["observation"] = action["observation"]
                    new_pre_actions.append(temp)
            process_item["pre_messages"] = pre_messages
            process_item["pre_actions"] = new_pre_actions

            action = {}
            action["thought"] = data["wrong_iteration"]["action"]["thought"]
            action["python_code"] = data["wrong_iteration"]["action"]["execute"].replace("```python", "").replace("```", "")
            action["json_function_calling"] = convert_to_json_function_calling(data["wrong_iteration"]["action"]["execute"].replace("```python", "").replace("```", ""))
            action["observation"] = data["wrong_iteration"]["observation"]
            process_item["action"] = action
            answer = {}
            answer["thought"] = data["wrong_iteration"]["reflection_iteration"]["action"]["thought"]
            answer["python_code"] = data["wrong_iteration"]["reflection_iteration"]["action"]["execute"].replace("```python", "").replace("```", "")
            answer["json_function_calling"] = convert_to_json_function_calling(data["wrong_iteration"]["reflection_iteration"]["action"]["execute"].replace("```python", "").replace("```", ""))
            process_item["answer"] = answer
            process_item["data_path"] = os.path.join(api_trace_dir, filename)
            api_trace_list.append(process_item)
        except Exception as e:
            print(f"Error: {e}")
            cnt += 1
            continue

print(cnt)
print(len(api_trace_list))
with open("data/RefineToolbench/data/I3.json", "w") as f:
    json.dump(api_trace_list, f, indent=4)