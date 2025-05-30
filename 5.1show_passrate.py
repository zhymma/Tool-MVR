import json
import os
from utils.utils import chat_completion
import random
from multiprocessing import Pool
import time
import tqdm
# base_path = "stabletoolbench/data_eval/answer/1231_test_lucky_tool_sft1_qwen2.5_20241231"
# base_path = "stabletoolbench/data_eval/answer/1231_test_lucky_tool_sft1_qwen2.5_20241230"
# base_path = "stabletoolbench/data_eval/answer/0101_test_lucky_tool_sft1_qwen2.5_20241231"
# base_path = "stabletoolbench/data_eval/answer/0101_test_lucky_tool_qwen2.5_instruct"
# base_path = "stabletoolbench/data_eval/answer/0103_test_lucky_tool_sft1_qwen2.5_20241231"
# base_path = "stabletoolbench/data_eval/answer/0103_test_lucky_tool_sft1_qwen2.5_20250102"
# base_path = "stabletoolbench/data_eval/answer/0104_test_qwen2.5_instruct_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0104_test_gpt4_instruct_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0104_test_gpt3.5_instruct_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0104_test_llama3.1_instruct_multi_agent"

# base_path = "stabletoolbench/data_eval/answer/0108_test_qwen2.5_sft_DFSFT"
# base_path = "stabletoolbench/data_eval/answer/0109_test_lucky_tool_sft1_qwen2.5_20250108_multi_agent"

# base_path = "stabletoolbench/data_eval/answer/0108_test_llama3.1_sft_DFSFT"
# base_path = "stabletoolbench/data_eval/answer/0108_test_gpt4_instruct_DFSDT"
# base_path = "stabletoolbench/data_eval/answer/0108_test_gpt3.5_instruct_DFSDT"
# base_path = "stabletoolbench/data_eval/answer/0109_test_lucky_tool_sft1_llama3.1_20250108_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0110_test_lucky_tool_sft2_llama3.1_20250110_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0112_test_lucky_tool_sft12_qwen2.5_20250111_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0112_test_lucky_tool_sft12_llama3.1_20250111_multi_agent"
# base_path = "stabletoolbench/data_eval/answer/0114_test_lucky_tool_sft12_qwen2.5_20250113_multi_agent"
base_path = "stabletoolbench/data_eval/answer/0121_test_llama3.1_instruct_DFSFT"
# base_path = "stabletoolbench/data_eval/answer/0121_test_qwen2.5_instruct_DFSFT"
print(base_path)

for group in [ "G1_instruction", "G1_category", "G1_tool", "G2_category", "G2_instruction", "G3_instruction"]:
# for group in [ "G2_category", "G2_instruction", "G3_instruction"]:
    folder_path = os.path.join(base_path, group)
    pass_rate = 0
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if "answer_status" in data:
                        if "answer_status" in data["answer_status"]:
                            result = data["answer_status"]["answer_status"]
                        elif "parse_answer_status" in data["answer_status"]:
                            result = data["answer_status"]["parse_answer_status"]
                        if result == "Pass" or result == "Solved":
                            pass_rate += 1
                        elif result == "Unsure":
                            pass_rate += 0.5
    print(group, pass_rate, len(os.listdir(folder_path)), pass_rate/len(os.listdir(folder_path)))
