from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.requests import Request
import uvicorn
import time
import json
import os, yaml
import requests
from typing import Union
from utils import standardize, change_name

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from tenacity import retry, wait_random_exponential, stop_after_attempt
import copy
import random
config_file='config.yml'
CONFIG = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
print(CONFIG)
CACHE_FOLDER = CONFIG['cache_folder']
# OpenAI API
from openai import OpenAI
if 'api_base' in CONFIG:
    OPENAI_API_BASE=CONFIG['api_base']
else:
    OPENAI_API_BASE="https://api.openai.com/v1"
OPENAI_API_KEY=CONFIG['api_key']

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class Info(BaseModel):
    category: str
    tool_name: str
    api_name: str
    tool_input: Union[str, dict]
    strip: str
    toolbench_key: str
    is_api_valid: bool

def prepare_tool_name_and_url(info):
    category = info.category
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")
    
    tool_name = info.tool_name
    # api_name = change_name(standardize(info.api_name)).split("_for_")[0]
    api_name = change_name(standardize(info.api_name)).split(f"_for_{tool_name}")[0]
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info.tool_name)
        code_string = f"""from my_tools.{standard_category}.{tool_name}.api import {api_name}"""
        tool_name += f"_for_{standard_category}"
    else:
        tmp_tool_name = standardize(tool_name.replace(f"_for_{standard_category}", ""))
        code_string = f"""from my_tools.{standard_category}.{tmp_tool_name}.api import {api_name}"""
    return tool_name, standard_category, api_name, code_string

@app.post('/virtual')
# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(1))
def get_virtual_response(request: Request, info: Info):
    user_key = info.toolbench_key
    is_function_valid = info.is_api_valid
    tool_name, standard_category, api_name, code_string = prepare_tool_name_and_url(info)
    tool_input = info.tool_input
    if type(tool_input) is dict:
        tool_input = json.dumps(tool_input)
    tool_name_original = info.tool_name
    if "```" in tool_input:
        tool_input = tool_input.replace("```json", "").replace("```", "").strip()
    tool_input = "{" + tool_input.strip("{}") + "}"
    if api_name == "chat_with_user":
        return {"error": "", "response": "Chat with user."}
    
    try:
        tool_input = json.loads(tool_input)
    except Exception as e:
        if tool_input == "":
            tool_input = {}
        elif isinstance(tool_input, dict):
            tool_input = tool_input
        else:
            print(f"Can not parse tool input into json: {tool_input}")
            print(type(tool_input))
            print(tool_input)
            response_dict = {"error": f"Tool input parse error...\n", "response": ""}
            return response_dict
    if not os.path.exists(CACHE_FOLDER):
        os.mkdir(CACHE_FOLDER)

    # load from cache
    cache = {}
    # prerequisite: to read files correctly, "my_tools_cache" folder and "toolenv/tools/" folder should be available
    try:
        if os.path.exists(os.path.join(CACHE_FOLDER, standard_category)):
            if os.path.exists(os.path.join(CACHE_FOLDER, standard_category, tool_name)):
                if os.path.exists(os.path.join(CACHE_FOLDER, standard_category, tool_name, api_name+".json")):
                    tools_cache_record = json.load(open(os.path.join(CACHE_FOLDER, standard_category, tool_name, api_name+".json"), "r"))
                    cache.update(tools_cache_record)
                    if str(tool_input) in cache:
                        # print("using cached real response")
                        response_dict = cache[str(tool_input)]
                        if 'is_fake' in response_dict:
                            is_fake_flag = response_dict.pop('is_fake')
                        else:
                            is_fake_flag = False

                        if not is_fake_flag and check_result(response_dict):
                            print("using cached response, is_fake:", is_fake_flag)
                            if "There was an error processing your API key" in response_dict['response']:
                                os._exit(0)
                            return response_dict
                        elif not is_fake_flag and not check_result(response_dict):
                            print("cached response is invalid, need to retry real response and fake response!")
                        # else:
                        #     print("using cached response, is_fake:", is_fake_flag)
                        #     return response_dict
    except Exception as e:
        print(f"Loading cache error: {e}")
    
    if is_function_valid:
        """
        Call the real api before generating fake response
        """
        
        headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        'toolbench_key': user_key
        }
        os.environ['HTTP_PROXY']= ''
        if "_for_" in tool_name_original:
            tool_name_real = tool_name_original.split("_for_")[0]
        else:
            tool_name_real = tool_name_original
        data = {
            "category": standard_category,
            "tool_name": tool_name_real,
            "api_name": api_name,
            "tool_input": tool_input,
            "strip": "",
            "toolbench_key": user_key
        }
        
        real_response = requests.post(CONFIG['toolbench_url'], headers=headers, data=json.dumps(data))

        print("real_response===========================================================================================================================")
        print(real_response.text)
        print("real_response===========================================================================================================================")
        # Check if the request was successful
        if real_response.status_code == 200:
            real_response = real_response.json() 
            if check_result(real_response):
                print("returning real_response")
                if CONFIG['is_save']:
                    save_cache(cache, tool_input, real_response, standard_category, tool_name, api_name, is_fake=False)
                    print("=====saved real_response=====")
                if "There was an error processing your API key" in real_response['response']:
                    os._exit(0)
                return real_response
            else:
                if 'error' in real_response['error'] and "missing 1 required positional argument" in real_response['error']:
                    if CONFIG['is_save']:
                        save_cache(cache, tool_input, real_response, standard_category, tool_name, api_name, is_fake=False)
                        print("=====saved real_response=====")
                    return real_response
        
    # os._exit(0)
    print("===========Need to generate Fake response============")
    if str(tool_input) in cache:
        response_dict = cache[str(tool_input)]
        if 'is_fake' in response_dict:
            is_fake_flag = response_dict.pop('is_fake')
        else:
            is_fake_flag = False
        if is_fake_flag:
            print("!!!!!!!!!!!!!!!!!!!!!!using cached fake response")
            if "There was an error processing your API key" in response_dict['response']:
                os._exit(0)
            return response_dict
    """
    Fake response function here. Use the cached history response for in-context examples.
    result = fake_response_function(api_doc, api_name, api_parameters, *kwargs)
    """

    # parse api_doc
    tool_name_original = standardize(tool_name_original)
    api_name = standardize(api_name)
    api_doc = {
        'tool_description': "",
        'api_info': "",
    }
    try:
        if os.path.exists(os.path.join(CONFIG['tools_folder'], standard_category)):
            if os.path.exists(os.path.join(CONFIG['tools_folder'], standard_category, tool_name_original.split("_for_")[0]+".json")):
                # read json
                api_intro = json.load(open(os.path.join(CONFIG['tools_folder'], standard_category, tool_name_original.split("_for_")[0]+".json"), "r"))
                # get tool_dexcription and api_info
                tool_description = api_intro['tool_description']
                api_info = []
                for api in api_intro['api_list']:
                    if api_name == standardize(api['name']):
                        api_info.append({
                            'name': api['name'],
                            'description': api['description']
                        })
                # check invalid api name
                if len(api_info) == 0:
                    print("cant match api name")
                api_doc = {
                    'tool_description': tool_description,
                    'api_info': api_info
                }
            else:
                print(f"cant get {tool_name_original}")
    except Exception as e:
        print(f"Loading api_doc error: {e}")
        print("###########Loading api_doc error, Exit!############")
        os._exit(0)

    # # get several examples from cache
    # example_num = 5
    # # get top example_num examples
    # api_example = list(cache.items())[:example_num]
    # while len(str(api_example)) > 2048 and example_num > 1:
    #     example_num -= 1
    #     api_example = list(cache.items())[:example_num]

    # print(f"api example: {api_example},,, tool_input: {tool_input},,, api_doc: {api_doc},")
    #! 不需要example
    api_example = ""

    result = fake_response_function_chat(api_example,tool_input,api_doc)
    print(f"========returning fake result==========")
    print(result)

    if CONFIG['is_save']:
        print(f"========saving fake result==========")
        save_cache(cache, tool_input, result, standard_category, tool_name, api_name, is_fake=True)

    if not isinstance(result, dict):
        res = json.loads(result)
        assert "is_fake" not in res
        return res
    else:
        assert "is_fake" not in result
        return result
    
def is_valid_json(result):
    """
    Checks if the given string is valid JSON.

    Args:
      data: The string to be checked.

    Returns:
      True if the string is valid JSON, False otherwise.
    """
    # check json format
    # return True
    try:
        result = json.loads(result)
        return True
    except Exception as e:
        print(f"Can not parse result into json: {result}")
        return False

def check_result(processes_value: dict):
    if 'error' not in processes_value or processes_value['error'] != '':
        return False
    if 'response' not in processes_value:
        return False
    response = str(processes_value['response'])
    if 'connection' in response.lower() or 'rate limit' in response.lower() or 'time out' in response.lower() or 'timed out' in response.lower() or 'does not exist' in response.lower() or 'internal error' in response.lower() or 'API doesn\'t exists' in response.lower() or "API doesn\'t exists" in response.lower() or response == '{\'message\': "API doesn\'t exists"}' or 'Service Not Found' in response:
        return False
    elif 'authoriz' in response.lower() or 'authenticat' in response.lower() or 'unauthorized' in response.lower() or 'blocked user' in response.lower() or 'unsubscribe' in response.lower() or 'blocked' in response.lower() or 'credential' in response.lower() or 'unauthenticated' in response.lower() or 'disabled for your subscription' in response.lower() or 'ACCESS_DENIED' in response:
        return False
    elif 'parse' in response.lower() or 'is not defined' in response.lower():
        return False
    elif 'DEPRECATED_ENDPOINT'.lower() in response.lower():
        return False
    elif 'invalid consumer key' in response.lower() or 'invalid' in response.lower():
        return False
    elif 'incorrect request' in response.lower() or 'incorrect' in response.lower():
        return False
    elif 'MISSING_ARG_ACCESS_TOKEN'.lower() in response.lower() or "API Key".lower() in response.lower():
        return False
    elif 'The captcha UUID has expired'.lower() in response.lower():
        return False
    elif len(response) == 0:
        return False
    elif "status_code=50" in response.lower() or "status_code=429" in response.lower() or "status_code" in response.lower() or "statusCode".lower() in response.lower() or "'success': False".lower() in response.lower() or "'data': []".lower() in response.lower():
        return False
    return True

# def check_result(processes_value: dict):
#     if 'error' not in processes_value or processes_value['error'] != '':
#         return False
#     if 'response' not in processes_value:
#         return False
#     response = str(processes_value['response'])
#     if 'http' in response.lower() or 'connection' in response.lower() or 'rate limit' in response.lower() or 'time out' in response.lower() or 'timed out' in response.lower() or 'does not exist' in response.lower() or '404' in response.lower() or '504' in response.lower() or '500' in response.lower() or 'internal error' in response.lower() or 'API doesn\'t exists' in response.lower() or "API doesn\'t exists" in response.lower() or response == '{\'message\': "API doesn\'t exists"}' or 'Service Not Found' in response:
#         return False
#     elif 'authoriz' in response.lower() or 'authenticat' in response.lower() or 'unauthorized' in response.lower() or 'blocked user' in response.lower() or 'unsubscribe' in response.lower() or 'blocked' in response.lower() or '401' in response.lower() or '403' in response.lower() or 'credential' in response.lower() or 'unauthenticated' in response.lower() or 'disabled for your subscription' in response.lower() or 'ACCESS_DENIED' in response:
#         return False
#     elif 'parameter' in response.lower() or 'parse' in response.lower() or 'is not defined' in response.lower():
#         return False
#     elif len(response) == 0:
#         return False
#     elif "status_code=50" in response or "status_code=429" in response:
#         return False
#     return True

def save_cache(cache, tool_input, result, standard_category, tool_name, api_name, save_folder=CACHE_FOLDER, is_fake=False):
    # save cache
    try:
        if isinstance(result, dict):
            cache[str(tool_input)] = copy.deepcopy(result)
            cache[str(tool_input)]['is_fake'] = is_fake
        elif isinstance(result, str):
            try:
                start_idx = result.find("{")
                end_idx = result.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    result = result[start_idx:end_idx+1]
                result_dict = json.loads(result)
                cache[str(tool_input)] = copy.deepcopy(result_dict)
                cache[str(tool_input)]['is_fake'] = is_fake
            except Exception as e:
                print(f"Load result failed: {e}")
                return

        if not os.path.exists(os.path.join(save_folder, standard_category)):
            os.mkdir(os.path.join(save_folder, standard_category))
        if not os.path.exists(os.path.join(save_folder, standard_category, tool_name)):
            os.mkdir(os.path.join(save_folder, standard_category, tool_name))    
        json.dump(cache, open(os.path.join(save_folder, standard_category, tool_name, api_name+".json"), "w"), indent=4)
    except Exception as e:
        print(f"Save cache failed: {e}")

def fake_response_function_chat(api_example, tool_input, api_doc):
    '''
    api_example: list of tuple, [(input, output), ...]
    tool_input: dict, input of the tool
    api_doc: dict, api document
    '''
    system_prompt = '''
Imagine you are an API Server operating within a specialized tool, which contains a collection of distinct APIs. Your task is to process the given input parameters and construct a meaningful, relevant, and structured JSON response based on these inputs and we provide API descriptions in the API documentation. Analyze the input carefully to understand its intended purpose, and generate the corresponding output data. Your response must follow the structure below:
```json
{
    "error": "",
    "response": "<Your_Response>"
}
```
Your response should be:
- Simple and concise, directly addressing the input.
- Well-structured and meaningful, providing relevant information based on the parameters.
- Rich and effective in content, ensuring that it adds value to the request.

If the provided input is incomplete or unclear, use your judgment to generate a practical, effective response. However, ensure that your output is still consistent with the intended function of the API and avoids unnecessary complexity.

Remember to follow the JSON format strictly, ensuring it is parsable and well-formed. And your response should be a valid json string in the format of ```json  ```.

---

This is an example of the response format:

API Documentation
API Name: GetFinalExamScores
Description: Retrieves detailed information about a student's final exam scores, including the student name, ID, subject, score, and grade based on the provided student ID or name.
Input Example:
```json
{
  "student_id": "12345",
  "student_name": "John Doe"
}
```
Output Example:
```json
{
  "error": "",
  "response": {
    "student_id": "12345",
    "student_name": "John Doe",
    "scores": [
      {
        "subject": "Mathematics",
        "score": 95,
        "grade": "A"
      }
    ],
    "average_score": 91.67,
    "overall_grade": "A"
  }
}
```

    '''
    system_prompt = {"role": "system", "content": system_prompt}
    # user prompt, truncated to 2048 characters if too long
    # user_prompt = "API Documentation:"+str(api_doc)+"\n"+"API Examples:"+str(api_example)[:2048]+"\n"+"API Input:"+str(tool_input)+"\n"
    user_prompt = "API Documentation:"+str(api_doc)+"API Input:"+str(tool_input)+"\n"+"API Output:"
    user_prompt = {"role": "user", "content": user_prompt}

    # client = OpenAI(
    #     api_key = OPENAI_API_KEY,
    #     base_url = OPENAI_API_BASE,
    # )
    max_retries = 6
    flag = False
    for attempt in range(max_retries):
        models = ["gpt-4-turbo", "gpt-4o", "gpt-4o-2024-05-13","gpt-4o-2024-11-20", "gpt-4","gpt-4-1106-preview"]
        model = random.choice(models)
        if attempt == 5:
            model = "gpt-4o-mini"
        response = requests.post(
            OPENAI_API_BASE,
            json={
                "key": OPENAI_API_KEY,
                "model": model,
                "messages": [system_prompt, user_prompt],
                "temperature": CONFIG['temperature'],
            },
        )
        
        if response.status_code == 200:
           
            result = response.json()
            # print(result)
            # print(type(result))
            if "error" in result:
                 print(f"Request failed on attempt {attempt + 1}, retrying...")
                 
                 time.sleep(random.randint(1, 5))
                 continue
            result = result["choices"][0]["message"]["content"]
            if "```json" in result:
                result = result.replace("```json", "").replace("```", "").strip()
            if is_valid_json(result):
                flag = True
                break
            print(f"Invalid JSON response on attempt {attempt + 1}. Retrying...")
        else:
            print(f"Request failed on attempt {attempt + 1}, retrying...")
        # time.sleep(1)

    if flag:
        return result
    else:
        fake_error = {
            "error": "Failed to generate fake response",
            "response": "",
        }
        return json.dumps(fake_error)

if __name__ == "__main__":
    uvicorn.run(app="main:app", host="127.0.0.1", port=CONFIG['port'])