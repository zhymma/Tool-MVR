import json
import random
import os
from multiprocessing import Pool
from utils.utils import chat_completion
import random
from multiprocessing import Pool


save_path_dir = (
    "data/RefineToolbench/Output/LLM_vllm_DFS_woFilter_w2_qwen2.5_instruct_depth_3"
)
refine_toolbench_path = "/data/user/code/luckytool/data/RefineToolbench/data"
I1_data = json.load(open(os.path.join(refine_toolbench_path, "I1.json"), "r"))
I2_data = json.load(open(os.path.join(refine_toolbench_path, "I2.json"), "r"))
I3_data = json.load(open(os.path.join(refine_toolbench_path, "I3.json"), "r"))


win_rate = {"I1": 0, "I2": 0, "I3": 0}
error_recognition_rate = {"I1": 0, "I2": 0, "I3": 0}
error_correction_rate = {"I1": 0, "I2": 0, "I3": 0}
for filename in os.listdir(save_path_dir):
    file_path = os.path.join(save_path_dir, filename)
    with open(file_path, "r") as f:
        data = json.load(f)
    group, id = filename.split(".")[0].split("@")

    if data["win"] == False:
        continue

    win_rate[group] += 1
    answer_status = data["result_status"]
    if (
        answer_status["error_recognition"] == "Pass"
        or answer_status["error_correction"] == "true"
        or answer_status["error_correction"] == True
    ):
        error_recognition_rate[group] += 1
    if (
        answer_status["error_correction"] == "Pass"
        or answer_status["error_correction"] == True
        or answer_status["error_correction"] == "true"
    ):
        error_correction_rate[group] += 1

print(f"win rate I1: {win_rate['I1']/len(I1_data)}, {win_rate['I1']}, {len(I1_data)}")
print(f"win rate I2: {win_rate['I2']/len(I2_data)}, {win_rate['I2']}, {len(I2_data)}")
print(f"win rate I3: {win_rate['I3']/len(I3_data)}, {win_rate['I3']}, {len(I3_data)}")

print(
    f"error recognition rate I1: {error_recognition_rate['I1']/len(I1_data)}, {error_recognition_rate['I1']}, {len(I1_data)}"
)
print(
    f"error recognition rate I2: {error_recognition_rate['I2']/len(I2_data)}, {error_recognition_rate['I2']}, {len(I2_data)}"
)
print(
    f"error recognition rate I3: {error_recognition_rate['I3']/len(I3_data)}, {error_recognition_rate['I3']}, {len(I3_data)}"
)

print(
    f"error correction rate I1: {error_correction_rate['I1']/len(I1_data)}, {error_correction_rate['I1']}, {len(I1_data)}"
)
print(
    f"error correction rate I2: {error_correction_rate['I2']/len(I2_data)}, {error_correction_rate['I2']}, {len(I2_data)}"
)
print(
    f"error correction rate I3: {error_correction_rate['I3']/len(I3_data)}, {error_correction_rate['I3']}, {len(I3_data)}"
)
