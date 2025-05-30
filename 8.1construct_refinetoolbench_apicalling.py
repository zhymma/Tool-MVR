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


def get_api_examples(api_doc_path, num_valid=500, num_invalid=500):
    """
    Get API examples from the API documentation file.
    Args:
        api_doc_path: Path to the API documentation file
        num_valid: Number of valid APIs to retrieve
        num_invalid: Number of invalid APIs to retrieve
    Returns:
        valid_apis: List of valid APIs with many parameters
        invalid_apis: List of invalid APIs with many parameters
    """
    with open(api_doc_path, "r") as f:
        api_doc = json.load(f)

    # First collect all APIs with 3+ parameters
    valid_candidates = []
    invalid_candidates = []

    for api_name, api_info in api_doc.items():
        # Check if API is valid
        is_valid = api_info.get("is_api_valid", False)

        # Get number of parameters
        try:
            num_params = len(api_info["api"]["function"]["parameters"]["properties"])
        except:
            continue
        required_params = len(api_info["api"]["function"]["parameters"]["required"])
        # Only consider APIs with 3+ parameters
        if num_params >= 3 and required_params >= 3:
            if is_valid:
                valid_candidates.append({"name": api_name, "info": api_info})
            else:
                invalid_candidates.append({"name": api_name, "info": api_info})

    # Randomly sample from candidates
    valid_apis = []
    invalid_apis = []

    if len(valid_candidates) > 0:
        valid_apis = random.sample(
            valid_candidates, min(num_valid, len(valid_candidates))
        )
    if len(invalid_candidates) > 0:
        invalid_apis = random.sample(
            invalid_candidates, min(num_invalid, len(invalid_candidates))
        )

    return valid_apis, invalid_apis


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


"""
### Example:

### Input (API Documentation):
  "get_timetable_for_flixbus_v2": {
    "api": {
      "type": "function",
      "function": {
        "name": "get_timetable_for_flixbus_v2",
        "description": "Retrieves the timetable for specified station and date. This function provides detailed schedule information, including timings, delays, cancellation status, and relevant geographical and directional data pertaining to each departure. Ensure the parameters such as station_id follows standard Flixbus UUID formats to avoid invalid parameter errors.",
        "parameters": {
          "type": "object",
          "properties": {
            "station_id": {
              "type": "string",
              "description": "The unique identifier for a Flixbus station, required in UUID format.",
              "example_value": "dcbd21fc-9603-11e6-9066-549f350fcb0c"
            },
            "date": {
              "type": "string",
              "description": "The date for which the timetable is requested, in the format DD.MM.YYYY.",
              "example_value": "15.05.2022"
            }
          },
          "required": [
            "station_id",
            "date"
          ],
          "optional": []
        }
      }
    },
    "is_api_valid": true
  },

### Output:
<query>I need to check the bus schedule from Berlin Central Station for January 15th, 2024. Can you help me find the timetable? The station ID for Berlin Central Station is "dcbd21fc-9603-11e6-9066-549f350fcb0c".</query>

<incorrect_api_call>
<thought>
I need to retrieve the timetable for Berlin Central Station. I'll use the get_timetable_for_flixbus_v2 API. 
</thought>
<execute>
```python
print(get_timetable_for_flixbus_v2(station_id="dcbd21fc-9603-11e6-9066-549f350fcb0c", date="2024-01-15"))
```
</execute>
</incorrect_api_call>
<correct_api_call_in_json>
{
  "api_name": "get_timetable_for_flixbus_v2",
  "parameters": {
    "station_id": "dcbd21fc-9603-11e6-9066-549f350fcb0c",
    "date": "15.01.2024"
  }
}
</correct_api_call_in_json>
"""
get_wrong_api_calling_prompt = """

You are an advanced AutoGPT agent, capable of utilizing multiple APIs to complete complex tasks. You will be provided with an API and documentation. Your task is to construct a test case for RefineToolbench (which test the ability of LLM to correct the incorrect API call). You need to generate a query, and solve the query using the provided APIs. However, you must deliberately introduce an incorrect API call during your reasoning process.

### Instructions:
1. **Query Generation**:
   - Based on the provided APIs, construct a complex query that requires API call to solve (in once iteration).
   - The query should be challenging enough to demonstrate proper API usage.
   - The query should be a real-world query in the XML format <query>query</query>.

2. **Incorrect API Call Format**:
   You must introduce an incorrect API call during your reasoning process to construct a test case for RefineToolbench.
   <incorrect_api_call>
   <thought>
   Explain reasoning for the API call, even though it will be incorrect but looks correct.
   </thought>
   <execute>
   ```python
   print(get_flight_prices(origin="Los Angeles", destination="Tokyo", date="2025-02-10"))
   ```
   </execute>
   </incorrect_api_call>
   The error type should be diverse based on the API documentation, such as
    - **API parameter errors**:
       - Missing required parameters.
       - Invalid input parameters.
       - Parameter type errors such as using string for integer parameter or wrong date format.
       - Parameter value exceeding the valid range.
       - Correctly formatted parameters but logically meaningless for the query.
    - **API name errors**:
       - Use wrong API name.
       - Use wrong API parameter.
       - Usage of non-existent APIs.
    The Execute code should be one line of code in python (without any comments).
    The Thought should be a sentence looks right, do not include any hints of the wrong action.
3. **Correct API Call Example**:
   - After the incorrect call, provide a JSON example of the correct API call in <correct_api_call_in_json> tag:
    <correct_api_call_in_json>
    {
      "api_name": "correct_api_name",
      "parameters": {
        "param1": "value1",
        "param2": "value2"
      }
    }
    </correct_api_call_in_json>
---

### Core Rules:
- **must** include:
  - <query>
  - <incorrect_api_call> It must has wrong API call.
  - <correct_api_call_in_json> It must has correct API call.
  
You should simulate a very weak LLM, which cannot understand the API parameters, to simulate its wrong cases in diverse ways.

---

### Notes:
1. Ensure that at least one incorrect API call is generated during reasoning.
2. The query should be diverse and the wrong API call should be hard to be corrected by LLM, because we want to test the ability of LLM to correct the incorrect API call !!!!!!
---

### Input (API Documentation):
===api_doc===
"""

api_doc_path = "data/process_data/all_apis/final_all_api.json"

# valid_apis, invalid_apis = get_api_examples(api_doc_path, num_valid=500, num_invalid=500)

with open("data/RefineToolbench/invalid_apis.json", "r") as f:
    invalid_apis = json.load(f)
with open("data/RefineToolbench/valid_apis.json", "r") as f:
    valid_apis = json.load(f)

print(len(valid_apis), len(invalid_apis))

import os

# Save valid and invalid APIs to RefineToolbench data folder
output_dir = "data/RefineToolbench"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save valid APIs
with open(f"{output_dir}/valid_apis.json", "w") as f:
    json.dump(valid_apis, f, indent=2)

# Save invalid APIs
with open(f"{output_dir}/invalid_apis.json", "w") as f:
    json.dump(invalid_apis, f, indent=2)


def process_api(api):
    try:
        api_name = api["function"]["name"]
        path = "data/RefineToolbench/saves/" + api_name + ".json"
        if os.path.exists(path):
            return 1
        prompt = get_wrong_api_calling_prompt.replace("===api_doc===", str(api))
        models = [
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-11-20",
            "gpt-4",
            "gpt-4-1106-preview",
        ]
        model = random.choice(models)
        response = chat_completion(
            key="xxx",
            messages=[{"role": "user", "content": prompt}],
            base_url="baseurl",
            temperature=0.9,
            model=model,
        )

        content = response["choices"][0]["message"]["content"]
        # Extract query
        query_start = content.find("<query>") + len("<query>")
        query_end = content.find("</query>")
        query = content[query_start:query_end].strip()

        # Extract incorrect API call
        incorrect_start = content.find("<incorrect_api_call>") + len(
            "<incorrect_api_call>"
        )
        incorrect_end = content.find("</incorrect_api_call>")
        incorrect_api_call = content[incorrect_start:incorrect_end].strip()

        # Extract thought and execute from incorrect API call
        thought_start = incorrect_api_call.find("<thought>") + len("<thought>")
        thought_end = incorrect_api_call.find("</thought>")
        thought = incorrect_api_call[thought_start:thought_end].strip()

        execute_start = incorrect_api_call.find("<execute>") + len("<execute>")
        execute_end = incorrect_api_call.find("</execute>")
        execute = incorrect_api_call[execute_start:execute_end].strip()

        # Extract correct API call JSON
        correct_start = content.find("<correct_api_call_in_json>") + len(
            "<correct_api_call_in_json>"
        )
        correct_end = content.find("</correct_api_call_in_json>")
        correct_api_call = json.loads(content[correct_start:correct_end].strip())

        result = {
            "query": query,
            "incorrect_api_call": {"thought": thought, "execute": execute},
            "correct_api_call": correct_api_call,
        }

        # Save result to JSON file
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        # Return all extracted components
        return result

    except Exception as e:
        print(e)
        return None


from multiprocessing import Pool, Manager


def process_api_wrapper(api):
    if "info" in api:
        result = process_api(api["info"]["api"])
    else:
        result = process_api(api)
    return result


if __name__ == "__main__":
    # manager = Manager()
    # correct_num = manager.Value('i', 0)
    # # Process valid APIs
    # with Pool(processes=8) as pool:
    #     results = pool.map(process_api_wrapper, valid_apis)
    #     for result in results:
    #         if result is not None:
    #             correct_num.value += 1

    # print(correct_num.value)
    # print(len(invalid_apis))
    # # Process invalid APIs
    # with Pool(processes=8) as pool:
    #     results = pool.map(process_api_wrapper, invalid_apis)
    #     for result in results:
    #         if result is not None:
    #             correct_num.value += 1
    #             print(result)
    # print(correct_num.value)

    # get the observation of the incorrect api call
    final_all_api = json.load(
        open("data/process_data/all_apis/final_all_api.json", "r")
    )
    all_api = json.load(open("data/process_data/all_apis/all_api.json", "r"))
    data_path = "data/RefineToolbench/APICallingSaves"

    # Iterate through API calling files
    def process_file(file_name):
        try:
            if not file_name.endswith(".json"):
                return

            file_path = os.path.join(data_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract API name from incorrect call
            if (
                "incorrect_api_call" not in data
                or "execute" not in data["incorrect_api_call"]
            ):
                return

            wrong_action = data["incorrect_api_call"]["execute"]
            if "```python" in wrong_action:
                wrong_action = wrong_action.replace("```python", "")
                wrong_action = wrong_action.replace("```", "")
            # Extract API name from the execute code
            api_name = file_name.split(".")[0]

            # Get code string from all_api
            code_string = all_api[api_name].get("code_string")
            code_string = code_string.replace(
                "http://localhost:8081/virtual", "http://localhost:8082/virtual"
            )

            is_api_valid = final_all_api[api_name].get("is_api_valid")
            add_string = '"is_api_valid": ' + str(is_api_valid) + ",\n"
            code_string = code_string.replace("payload = {", "payload = {" + add_string)
            code = code_string + "\n" + wrong_action
            result = execute_code(code)
            data["incorrect_api_call"]["observation"] = result
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            print(api_name, result)
            print("save to file", file_path)
        except Exception as e:
            print(e)
            return

    # Create process pool and process files in parallel
    with Pool(processes=8) as pool:
        list(
            tqdm(
                pool.imap(process_file, os.listdir(data_path)),
                total=len(os.listdir(data_path)),
            )
        )
