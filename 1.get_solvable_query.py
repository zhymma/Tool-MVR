import json
import yaml
from pathlib import Path
from typing import List, Dict
from utils.utils import chat_completion
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Event
import asyncio
import aiohttp
import os
from queue import Queue
from datetime import datetime, timedelta
from multiprocessing import Pool
def load_config() -> Dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_instructions(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return json.load(f)

def create_solvable_check_prompt(query: str, available_tools: List[Dict] = None) -> Dict:
    tools_str = f"{json.dumps(available_tools)}" if available_tools else ""
    
    prompt = """
You are an assistant responsible for evaluating whether a given user query can be solved using the provided APIs and tools. Your evaluation must consider both the completeness of the query's information, the capabilities of the available APIs, and the overall quality of the query. Follow the rules and examples below to determine solvability and assess the quality of the query.

---

### **Rules**

1. **Invalid Information:**  
   If the query contains invalid or nonsensical information (e.g., an invalid email address, phone number, or malformed input), return **"Unsolvable"**.

2. **Missing Information:**  
   If the query lacks essential information required to solve it (e.g., a restaurant name for a navigation task or a missing date for a scheduling task), return **"Unsolvable"**.

3. **Incomplete or Ambiguous Tools:**  
   If you are unable to determine how to solve the query with the provided tools, return **"Unsolvable"**.

4. **Solvable Query:**  
   If the query provides sufficient valid information and the available APIs/tools can solve it (even if some parameter values are implicit or assumed), return **"Solvable"**.

---

### **Query Quality Scoring (1 to 10)**

In addition to evaluating solvability, assess the **quality of the query** based on the following factors:

1. **Solvability:** How easy it is to solve the query using the provided tools? (Higher score if solvable, lower if unsolvable)  
2. **Semantic Clarity:** Is the query clear and grammatically correct? (Higher score for precise and clear queries)  
3. **Information Completeness:** Does the query contain all the necessary details? (Higher score for detailed and complete queries)  
4. **Reasoning Difficulty:** How complex is the reasoning needed to solve the query? (Higher score for queries with minimal ambiguity)  
5. **Tool Compatibility:** How well does the query align with the capabilities of the provided tools? (Higher score if tools can handle the query efficiently)

Assign a quality score from 1 to 10, where **10 represents a well-formed, solvable, and clear query** and **1 represents a poorly-formed, unsolvable query.**

---

### **Input Details**

You will be given:  
1. A **query**: This is the user's request.  
2. A list of **available tools**: These include APIs or functions, each described by their domain, name, description, required parameters, and output fields.  

---

### **Steps to Evaluate Solvability and Quality**

1. Analyze the **query** to determine whether it provides all the necessary and valid information. Missing or invalid information renders the query **Unsolvable**.  
2. Match the **query's requirements** against the **available tools** to see if the tools can provide the requested solution. Multiple tools may be combined if needed.  
3. Assess the query's **quality score** based on the criteria provided above.  
4. Use the rules above to provide a decision ("Solvable" or "Unsolvable").  

---

### **Output Format**

- **Explanation:** Provide a clear and concise explanation for your decision.  
- **Answer:**```json
{
    "decision": "Solvable" or "Unsolvable",
    "quality_score": 1 to 10
}
``` 

---

### **Examples**

#### Example 1:  
**Query:** "Can you fetch the flight data for the company AZU on June 15th, 2022?"  
**Available Tools:**  
- **Tool 1** - Domain: Flights. Name: get_flight_data.  
  Description: Fetch flight data by company and date.  
  Parameters: company, date.  
  Output: flight_information.

**Evaluation:**  
- The query provides all required parameters (`company` = AZU, `date` = June 15th, 2022).  
- The available tool (`get_flight_data`) can handle the query directly.  

**Explanation:** The query provides complete and valid information, and the available tool can fully address it.  
**Answer:**
```json
{
    "decision": "Solvable",
    "quality_score": 7
}
``` 

---

#### Example 2:  
**Query:** "Can you fetch the flight data for the company?"  
**Available Tools:**  
- **Tool 1** - Domain: Flights. Name: get_flight_data.  
  Description: Fetch flight data by company and date.  
  Parameters: company, date.  
  Output: flight_information.

**Evaluation:**  
- The query is missing the `date` parameter, which is required by the API.  

**Explanation:** The query does not provide all required parameters (`date` is missing), so it cannot be solved.  
**Answer:**
```json
{
    "decision": "Unsolvable",
    "quality_score": 2
}
``` 

---

#### Example 3:  
**Query:** "I'm traveling with my family. Can you tell us what's the weather like in Lisbon for tomorrow? Also, prepare for us an itinerary."  
**Available Tools:**  
- **Tool 1** - Domain: Weather. Name: get_weather.  
  Description: Get weather by city and date.  
  Parameters: city, date.  
  Output: weather_description.  
- **Tool 2** - Domain: Travel. Name: prepare_itinerary.  
  Description: Prepare an itinerary.  
  Parameters: city, duration.  
  Output: places_list.

**Evaluation:**  
- The query mentions "Lisbon" as the city and "tomorrow" as the date, which can be assumed implicitly.  
- The `get_weather` API can handle the weather request with `city` and `date`.  
- The `prepare_itinerary` API can handle the second request, even though the `duration` is not explicitly mentioned (non-specific parameters are allowed).

**Explanation:** The query provides sufficient information, and the available tools can solve both parts of the request.  
**Answer:**
```json
{
    "decision": "Solvable",
    "quality_score": 8
}
``` 

---

#### Example 4:  
**Query:** "Can you send a confirmation email to john_doe@example?"  
**Available Tools:**  
- **Tool 1** - Domain: Communication. Name: send_email.  
  Description: Send an email to a recipient.  
  Parameters: email_address, subject, body.  
  Output: email_status.

**Evaluation:**  
- The email address provided (`john_doe@example`) is invalid.  

**Explanation:** The query contains invalid information (malformed email address), making it unsolvable.  
**Answer:**
```json
{
    "decision": "Unsolvable",
    "quality_score": 1
}
``` 

---

Now, It's your turn to evaluate the following query:  
**Query:** ===query===  
**Available Tools:**  
===tools===  
"""
    prompt = prompt.replace("===query===", query).replace("===tools===", tools_str)
    return [{"role": "user", "content": prompt}]

import json
import yaml
from pathlib import Path
from typing import List, Dict
import requests
import os

def load_config() -> Dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_instructions(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return json.load(f)


def process_query(config: Dict, query_data: Dict, output_path: str):
    import time
    import random
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # print(f"Post a request to  for {query_data['query_id']}")
            response = requests.post(
                config["base_url"],
                json={
                    "key": config["api_key"],
                    "model": "gpt-4o-mini",
                    "messages": create_solvable_check_prompt(query_data["query"]),
                    "temperature": 0.7,
                },
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    raise Exception(result["error"])
                break
            else:
                # 如果请求失败，等待随机时间后重试
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = random.uniform(1, 10)
                    print(f"Request failed, retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    continue
                result = {
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = random.uniform(1, 10)
                print(f"Error occurred: {e}, retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                continue
            print(f"Error for query {query_data['query_id']}: {e}")
            return
    
    result_data = {
        "query_id": query_data["query_id"],
        "result": result
    }
    
    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2)



if __name__ == "__main__":
    config = load_config()
    instruction_files = [
        "data/instruction/G1_query.json",
        "data/instruction/G3_query.json",
       "data/instruction/G2_query.json",
    ]
    
    #! 标注query的质量
    # for idx, file_path in enumerate(instruction_files):
    #     file_name = file_path.split("/")[-1].split(".")[0]
    #     type = file_name.split("_")[0]
        
    #     # Load winning ids
    #     is_winning = f"data/process_data/{type}_winning_query_ids.json"
    #     with open(is_winning, "r") as f:
    #         winning_ids = json.load(f)
        
    #     instructions = load_instructions(file_path)
        
    #     usable_instructions = []
    #     for query_data in instructions:
    #         output_path = f"data/process_data/solvable_query_results/{type}/{query_data['query_id']}.json"
    #         output_path2 = f"data/process_data/solvable_query_results/{type}/{query_data['query_id']}_gpt4-turbo.json"
    #         # old_path = f"data/process_data/solvable_query_results/{type}/{query_data['query_id']}_4o_mini.json"
    #         # if os.path.exists(old_path):
    #         #     os.rename(old_path, output_path)
    #         if query_data["query_id"] in winning_ids and not os.path.exists(output_path) and not os.path.exists(output_path2):
    #             usable_instructions.append(query_data)
    #     # 随机打乱usable_instructions
    #     import random
    #     random.shuffle(usable_instructions)
    #     print(f"Processing {len(usable_instructions)} queries in {type}")
    #     def process_query_wrapper(args):
    #         config, query_data, type = args
    #         output_path = f"data/process_data/solvable_query_results/{type}/{query_data['query_id']}_gpt4-turbo.json"
    #         try:
    #             process_query(config, query_data, output_path)
    #             print(f"Task {query_data['query_id']} completed")
    #         except Exception as e:
    #             print(f"Task failed: {e}")
        
    #     # Create process pool
    #     with Pool(processes=100) as pool:
    #         # Prepare arguments for each process
    #         args = [(config, query_data, type) for query_data in usable_instructions]
    #         # Execute processes
    #         pool.map(process_query_wrapper, args)
        
    #     print(f"Results for {file_name} processed in {type}.")

    #！统计query的质量和solvable的query数量
    # Analyze results for each group

    
    # for group in ['G1', 'G2', 'G3']:
    #     result_dir = f'data/process_data/solvable_query_results/{group}'
    #     if not os.path.exists(result_dir):
    #         continue
            
    #     total_queries = 0
    #     solvable_ids = []
    #     all_quality = {}
        
    #     # Process each result file
    #     for file_name in os.listdir(result_dir):
    #         if not file_name.endswith('.json'):
    #             continue
                
    #         file_path = os.path.join(result_dir, file_name)
    #         id = file_name.split(".")[0]
    #         if "_" in id:
    #             id = id.split("_")[0]
    #         with open(file_path, 'r') as f:
    #             try:
    #                 data = json.load(f)
    #                 if 'error' in data:
    #                     os.remove(file_path)
    #                     continue
    #                 if 'result' in data:
    #                     result_content = data['result']['choices'][0]['message']['content']
    #                 else:
    #                     result_content = data['choices'][0]['message']['content']
                    
    #                 # Extract decision and score using basic string parsing
    #                 if '"decision": "Solvable"' in result_content:
    #                     solvable_ids.append(id)
                    
    #                 # Find quality score
    #                 score_start = result_content.find('"quality_score": ') 
    #                 if score_start != -1:
    #                     score_end = result_content.find('\n', score_start)
    #                     score_str = result_content[score_start:score_end]
    #                     try:
    #                         quality_score = int(score_str.split(':')[1].strip().rstrip(',}'))
    #                         all_quality[id] = quality_score
    #                     except:
    #                         continue
                            
    #                 total_queries += 1
                    
    #             except Exception as e:
    #                 print(f"Error processing {file_path}: {e}")
    #                 continue
        
    #     # Print statistics
    #     if total_queries > 0:
    #         print(f"\nResults for {group}:")
    #         print(f"Total queries analyzed: {total_queries}")
    #         print(f"Solvable queries: {len(solvable_ids)} ({(len(solvable_ids)/total_queries)*100:.1f}%)")
    #         print(f"Average quality score: {sum(all_quality.values())/len(all_quality):.2f}")
        
    #     # 保存结果
    #     with open(f"data/process_data/solvable_query_results/{group}_solvable_ids.json", "w") as f:
    #         json.dump(solvable_ids, f, indent=2)
    #     with open(f"data/process_data/solvable_query_results/{group}_quality.json", "w") as f:
    #         json.dump(all_quality, f, indent=2)

        
    # Process each group to find high quality IDs
    # for group in ['G1', 'G2', 'G3']:
    #     # Load quality scores and solvable IDs
    #     with open(f"data/process_data/solvable_query_results/{group}_quality.json", "r") as f:
    #         quality_scores = json.load(f)
            
    #     with open(f"data/process_data/solvable_query_results/{group}_solvable_ids.json", "r") as f:
    #         solvable_ids = json.load(f)
            
    #     # Find IDs with scores 8-10 that are also solvable
    #     high_quality_ids = []
        
    #     for query_id, score in quality_scores.items():
    #         # todo 8-10
    #         if score in [10] and query_id in solvable_ids:
    #             high_quality_ids.append(str(query_id))
    #     high_quality_ids = list(set(high_quality_ids))
    #     print(f"\nHigh quality solvable queries in {group}:")
    #     print(len(high_quality_ids))
        
    #     # Save results
    #     with open(f"data/process_data/solvable_query_results/{group}_high_quality_ids.json", "w") as f:
    #         json.dump(high_quality_ids, f, indent=2)

    # Process each group to create filtered query files
    # for group in ['G1', 'G2', 'G3']:
    #     # Load high quality IDs
    #     with open(f"data/process_data/solvable_query_results/{group}_high_quality_ids.json", "r") as f:
    #         high_quality_ids = json.load(f)
        
    #     # Load original queries
    #     with open(f"data/instruction/{group}_query.json", "r") as f:
    #         train_queries = json.load(f)
        
    #     if group == "G1":
    #         with open(f"stabletoolbench/solvable_queries/test_instruction/G1_category.json", "r") as f:
    #             test_queries1 = json.load(f)
    #         with open(f"stabletoolbench/solvable_queries/test_instruction/G1_instruction.json", "r") as f:
    #             test_queries2 = json.load(f)
    #         with open(f"stabletoolbench/solvable_queries/test_instruction/G1_tool.json", "r") as f:
    #             test_queries3 = json.load(f)
    #         queries = test_queries1 + test_queries2 + test_queries3
    #     elif group == "G2":
    #         with open(f"stabletoolbench/solvable_queries/test_instruction/G2_category.json", "r") as f:
    #             test_queries1 = json.load(f)
    #         with open(f"stabletoolbench/solvable_queries/test_instruction/G2_instruction.json", "r") as f:
    #             test_queries2 = json.load(f)
    #         queries = test_queries1 + test_queries2
    #     elif group == "G3":
    #         with open(f"stabletoolbench/solvable_queries/test_instruction/G3_instruction.json", "r") as f:
    #             test_queries = json.load(f)
    #         queries = test_queries
        
    #     test_ids = [str(query['query_id']) for query in queries]

    #     # high_quality_ids = [id for id in high_quality_ids if id not in test_ids]  
    #     high_quality_ids = [id for id in high_quality_ids]
    #     high_quality_ids = high_quality_ids + test_ids
    #     high_quality_ids = list(set(high_quality_ids))
    #     # Filter queries to only include high quality ones
    #     filtered_queries = [query for query in train_queries if str(query['query_id']) in high_quality_ids]
        
    #     # Save filtered queries
    #     with open(f"data/instruction/final_{group}_query.json", "w") as f:
    #         json.dump(filtered_queries, f, indent=2)
            
    #     print(f"\nSaved {len(filtered_queries)} filtered queries for {group}")

    # Analyze score distribution for each group
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    
    # plt.figure(figsize=(15, 5))
    
    # for i, group in enumerate(['G1', 'G2', 'G3'], 1):
    #     # Load quality scores
    #     try:
    #         with open(f"data/process_data/solvable_query_results/{group}_quality.json", "r") as f:
    #             quality_scores = json.load(f)
                
    #         # Count frequency of each score
    #         score_distribution = {}
    #         for score in range(1, 11):
    #             score_distribution[score] = len([q_id for q_id, q_score in quality_scores.items() if q_score == score])
            
    #         # Create subplot for this group
    #         plt.subplot(1, 3, i)
    #         bars = plt.bar(score_distribution.keys(), score_distribution.values(), 
    #                       color=['#0047AB' if score < 8 else '#D4AF37' for score in range(1, 11)])
            
    #         # Add value labels on top of each bar
    #         for bar in bars:
    #             height = bar.get_height()
    #             plt.text(bar.get_x() + bar.get_width()/2., height,
    #                     f'{int(height)}',
    #                     ha='center', va='bottom')
                        
    #         plt.title(f'Query Quality Score Distribution for {group}')
    #         plt.xlabel('Score')
    #         plt.ylabel('Count')
            
    #         # Calculate statistics
    #         total_queries = len(quality_scores)
    #         avg_score = sum([score for score in quality_scores.values()]) / total_queries
            
            
    #     except FileNotFoundError:
    #         print(f"\nNo quality scores file found for {group}")
    #         continue
    
    # plt.tight_layout()
    # plt.show()
    # # Save the plot
    # plt.savefig("data/process_data/solvable_query_results/quality_score_distribution.pdf",bbox_inches='tight')
    # # save

    # Analyze unsolvable queries
    with open("data/instruction/G1_query.json", "r") as f:
        all_queries = json.load(f)
        
    with open("data/process_data/solvable_query_results/G1_solvable_ids.json", "r") as f:
        solvable_ids = json.load(f)
    unsolvable_queries = []
    try:
        for item in all_queries:
            qid = str(item["query_id"])
            if qid not in solvable_ids:
                unsolvable_queries.append(item["query"])

        # Print 5 random unsolvable queries as examples
        import random
        if unsolvable_queries:
            samples = unsolvable_queries[:10]
            print("\n\n".join(samples))
        else:
            print("No unsolvable queries found")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
