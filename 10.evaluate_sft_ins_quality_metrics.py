import json
from utils.utils import chat_completion
import os
import random

save_path = "data/training_data/evaluate_quality"

training_data_path = "data/training_data/0108_10_training_data.json"
with open(training_data_path, "r") as f:
    training_data = json.load(f)


specificity_prompt = """
You are tasked with evaluating whether a user instruction provides all necessary information.
(Some information might be provided in the query, some might be provided in the API Calling Result)
Please analyze the query and API documentation to determine if all required parameters are explicitly provided.

Answer with only "Yes" if all necessary information are provided, or "No" if any required information is missing.

Example 1:
Query: "Can you fetch the flight data for the company AZU on June 15th, 2022?"
Required Parameters: company and date
Answer: Yes

Example 2:
Query: "Can you fetch the flight data for the company?" 
Answer: No

---

Query: ===query===
API Doc: ===api_doc===

Answer:
"""

from multiprocessing import Pool


def process_specificity(data):

    id = data["id"]

    backup_path = f"{save_path}/{id}.json"
    if os.path.exists(backup_path):
        with open(backup_path, "r") as f:
            metric = json.load(f)
        if "specificity" in metric:
            print(id, metric["specificity"])
            return metric["specificity"]
    else:
        metric = {}
    group = id.split("@")[0]
    qid = id.split("@")[1]
    file_path = f"stabletoolbench/data_eval/answer/new_train_0106/{group}/{qid}_multi_agent.json"
    with open(file_path, "r") as f:
        item = json.load(f)
    query = item["answer_generation"]["query"]
    api_doc = str(item["answer_generation"]["function"])
    prompt = specificity_prompt.replace("===query===", query).replace(
        "===api_doc===", api_doc
    )
    response = chat_completion(
        "xxx",
        [{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
        base_url="baseurl",
        temperature=0.0,
    )
    if "error" in response:
        print(response)
        return False
    result = response["choices"][0]["message"]["content"]
    if "Answer: No" in result:
        specificity = False
    else:
        specificity = True
    metric["specificity"] = specificity
    with open(backup_path, "w") as f:
        json.dump(metric, f)
    print(id, specificity)
    return specificity


coherence_prompt = """
You are tasked with assessing the overall coherence of a query.
A coherent query has a clear logical flow and all parts are related to each other.
Analyze the query and determine if it is Coherent or Incoherent.

Examples:
Query:
"I'm a cryptocurrency trader, and I want to analyze the historical prices and market caps of popular cryptocurrencies like Bitcoin, Ethereum, and Stellar. Can you fetch this information for me using the Crypto Prices API? Additionally, I'm planning a trip to North America and I would like to know the subregions in North America using the Geography API."

The query starts about crypto analysis but then switches to an unrelated geography topic
Answer: Incoherent

---

Query:
"I want to impress my friends with some hilarious jokes. Can you fetch some Chuck Norris jokes related to 'sports'? Also, fetch me some Chuck Norris jokes related to 'art'?"

All parts relate to fetching Chuck Norris jokes
Answer: Coherent

---

Query: 
===query===

Answer:

"""


def process_coherence(data):
    id = data["id"]
    group = id.split("@")[0]
    qid = id.split("@")[1]
    file_path = f"stabletoolbench/data_eval/answer/new_train_0106/{group}/{qid}_multi_agent.json"
    backup_path = f"{save_path}/{id}.json"
    if os.path.exists(backup_path):
        with open(backup_path, "r") as f:
            metric = json.load(f)
            if "coherence" in metric:
                print(id, metric["coherence"])
                return metric["coherence"]
    else:
        metric = {}
    with open(file_path, "r") as f:
        item = json.load(f)
    query = item["answer_generation"]["query"]
    prompt = coherence_prompt.replace("===query===", query)
    response = chat_completion(
        "xxx",
        [{"role": "user", "content": prompt}],
        "baseurl",
        temperature=0.0,
        model="gpt-4o-mini",
    )
    if "error" in response:
        print(response)
        return False
    result = response["choices"][0]["message"]["content"]
    if "Answer: Incoherent" in result:
        coherence = False
    else:
        coherence = True
    metric["coherence"] = coherence
    with open(backup_path, "w") as f:
        json.dump(metric, f)
    print(id, coherence)
    return coherence


solvability_prompt = """
You will be given a user query and a list of API functions that can return external information. Each API function is shown with its domain, name, description, parameters and output fields.

Determine whether any subset of the API functions could provide all the information required to answer the query. No need for one API to answer all the requests in the query, multiple APIs can be used to answer different requests in the query. It is alright if a parameter value is not explicitly provided in the query. Use the information provided in Domain, Name, Description, and parameters.

Answer with 'Yes' or 'No' and provide an explanation for your answer.

Example 1:
Query: "I'm traveling with my family. Can you tell us what's the weather like in Lisbon for tomorrow? Also, prepare for us an itinerary"

API Function 1
Domain: Weather
Name: get_weather
Description: get weather by city and date
Parameters: city, date
Output: weather description

API Function 2 
Domain: Traveling
Name: prepare_itinerary
Description: prepare itinerary
Parameters: city, duration
Output: places list

Explanation: The get_weather function provides the weather given a city and date. We do not know the date of tomorrow, but we allow non-explicit parameter values. The prepare_itinerary can handle the second request. Duration is not mentioned, but we allow non-specific parameters.

Answer: Yes

---

Query: ===query===
API Functions: ===api_functions===

Answer:
"""


def process_solvability(data):
    id = data["id"]
    backup_path = f"{save_path}/{id}.json"
    group = id.split("@")[0]
    qid = id.split("@")[1]
    file_path = f"stabletoolbench/data_eval/answer/new_train_0106/{group}/{qid}_multi_agent.json"

    if os.path.exists(backup_path):
        with open(backup_path, "r") as f:
            metric = json.load(f)
            if "solvability" in metric:
                print(id, metric["solvability"])
                return metric["solvability"]
    else:
        metric = {}

    with open(file_path, "r") as f:
        item = json.load(f)
    query = item["answer_generation"]["query"]
    api_functions = str(item["answer_generation"]["function"])
    prompt = solvability_prompt.replace("===query===", query).replace(
        "===api_functions===", api_functions
    )
    response = chat_completion(
        "xxx",
        [{"role": "user", "content": prompt}],
        "baseurl",
        temperature=0.0,
        model="gpt-4o-mini",
    )
    if "error" in response:
        print(response)
        return False
    result = response["choices"][0]["message"]["content"]
    if "Answer: No" in result:
        solvability = False
    else:
        solvability = True

    metric["solvability"] = solvability
    with open(backup_path, "w") as f:
        json.dump(metric, f)
    print(id, solvability)
    return solvability


sufficiency_minimality_prompt = """
You will be given a user query and a list of API functions that are suggested to address the user requests, only a subset of them is relevant to answer the query. Each API function is shown with its domain, name, description, parameters and output fields.

Determine whether the API call sequences functionality solves the user query, and whether it solves it with minimal number of calls or it has redundant calls that are not needed to solve the query. Use the information provided in Domain, Name, Description, and parameters. If Description not provided, consider the API as not solving.

Each query has multiple requests. The API call sequence should address all different requests. Answer with two steps:
- calls_solves: (Yes/No) whether the API call sequence would fully solve the user query
- minimal_calls: (Yes/No) whether the API call sequence solves the user query with minimal number of calls (Yes) or it has redundant calls that are not needed to solve the query (No)

Example 1:
Query: "Can you fetch random fashion images. Then, create a video from these images"
API call sequence: [get_random_image(query='fashion'), get_video()]

All APIs Documentation:
API Function 1
- Domain: Content
- Name: get_random_image
- Description: get list of images from the internet
- Parameters: query
- Output: image

API Function 2 
- Domain: Content
- Name: get_video
- Description: get trending videos
- Parameters: query
- Output: video

Explanation: get_random_image fetches random images. get_video does not handle the second request of creating video.
calls_solves: No
minimal_calls: No

Example 2:
Query: "Fetch me latest news about NBA"
API call sequence: [get_news(), fetch_news()]

API Function 2
- Domain: NBA
- Name: get_news
- Description: get latest news
- Parameters: None
- Output: text


calls_solves: Yes
minimal_calls: Yes

---

Query: ===query===
API Functions: ===api_functions===
API call sequence: ===api_call_sequence===

Answer:
"""


def process_sufficiency_minimality(data):
    id = data["id"]

    backup_path = f"{save_path}/{id}.json"
    if os.path.exists(backup_path):
        with open(backup_path, "r") as f:
            metric = json.load(f)
        if "sufficiency" in metric and "minimality" in metric:
            print(id, metric["sufficiency"], metric["minimality"])
            return metric["sufficiency"], metric["minimality"]
    else:
        metric = {}

    group = id.split("@")[0]
    qid = id.split("@")[1]
    file_path = f"stabletoolbench/data_eval/answer/new_train_0106/{group}/{qid}_multi_agent.json"
    with open(file_path, "r") as f:
        item = json.load(f)

    query = item["answer_generation"]["query"]
    api_functions = str(item["answer_generation"]["function"])
    api_call_sequence = str(item["answer_generation"]["actions"])
    prompt = (
        sufficiency_minimality_prompt.replace("===query===", query)
        .replace("===api_functions===", api_functions)
        .replace("===api_call_sequence===", api_call_sequence)
    )

    response = chat_completion(
        "xxx",
        [{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
        base_url="baseurl",
        temperature=0.0,
    )
    if "error" in response:
        print(response)
        return False
    result = response["choices"][0]["message"]["content"]
    sufficiency = "calls_solves: No" not in result
    minimality = "minimal_calls: No" not in result

    metric["sufficiency"] = sufficiency
    metric["minimality"] = minimality
    with open(backup_path, "w") as f:
        json.dump(metric, f)

    print(id, sufficiency, minimality)
    return sufficiency, minimality


if __name__ == "__main__":
    # all_specificity = []
    # with Pool(processes=30) as pool:
    #     all_specificity = pool.map(process_specificity, training_data)
    # # print(all_specificity)
    # all_coherence = []
    # with Pool(processes=30) as pool:
    #     all_coherence = pool.map(process_coherence, training_data)

    # all_solvability = []
    # with Pool(processes=30) as pool:
    #     all_solvability = pool.map(process_solvability, training_data)
    # # print(all_solvability)
    # all_sufficiency_minimality = []
    # with Pool(processes=30) as pool:
    #     all_sufficiency_minimality = pool.map(process_sufficiency_minimality, training_data)
    # print(all_sufficiency_minimality)

    # Calculate average metrics from evaluation files
    eval_dir = "data/training_data/evaluate_quality"
    total_metrics = {
        "specificity": 0,
        "coherence": 0,
        "solvability": 0,
        "specificity_coherence_solvability": 0,
        "sufficiency": 0,
        "minimality": 0,
        "sufficiency_minimality": 0,
    }
    count = 0

    for filename in os.listdir(eval_dir):
        if filename.endswith(".json"):
            with open(os.path.join(eval_dir, filename), "r") as f:
                metrics = json.load(f)
                for metric in total_metrics:
                    if metric in metrics:
                        total_metrics[metric] += 1 if metrics[metric] else 0
                if (
                    metrics["specificity"]
                    and metrics["coherence"]
                    and metrics["solvability"]
                ):
                    total_metrics["specificity_coherence_solvability"] += 1
                if metrics["sufficiency"] and metrics["minimality"]:
                    total_metrics["sufficiency_minimality"] += 1
            count += 1

    # Calculate and print averages
    print("\nMetrics Averages:")
    print("-----------------")
    for metric in total_metrics:
        avg = total_metrics[metric] / count if count > 0 else 0
        print(f"{metric}: {avg:.3f}")
    print(f"\nTotal evaluated files: {count}")
