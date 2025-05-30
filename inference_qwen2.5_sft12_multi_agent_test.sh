cd stabletoolbench
export PYTHONPATH=./
export VLLM_API_BASE="http://0.0.0.0:8002/v1"   # the address of vllm.server
export SERVICE_URL="http://localhost:8081/virtual"  # the address of api server
export MODEL_PATH="qwen2" # the name of vllm.server
export STRATEGY="tool_mvr"  # or CoT@1

export OUTPUT_DIR="data_eval/answer/tool_mvr"

for group in G2_category G2_instruction G3_instruction; do
    mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
    python toolbench/inference/qa_pipeline_multithread.py \
        --backbone_model LLM_vllm \
        --chatgpt_model "gpt-4o"\
        --model_path ${MODEL_PATH} \
        --max_observation_length 1024 \
        --single_chain_max_step 12 \
        --method ${STRATEGY} \
        --input_query_file solvable_queries/test_instruction/new_${group}.json \
        --output_answer_file $OUTPUT_DIR/$group \
        --max_query_count 30 \
        --num_thread 2
done