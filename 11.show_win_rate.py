import json
import os
from utils.utils import chat_completion
import random
from multiprocessing import Pool
import time
import tqdm

base_path = "stabletoolbench/data_eval/answer/0114_test_lucky_tool_sft12_llama3.1_20250113_multi_agent"



print(base_path)

for group in [ "G1_instruction", "G1_category", "G1_tool", "G2_category", "G2_instruction", "G3_instruction"]:
    folder_path = os.path.join(base_path, group)
    win_rate = 0
    total_count = 0
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if "win_rate_status_toolllama3.1" in data:
                        if data["win_rate_status_toolllama3.1"] == "Pass":
                            win_rate += 1
                        total_count += 1
    print(group, win_rate, total_count, win_rate/total_count)