executable = baseline_inference.sh
getenv     = true
arguments  = --model_dir MODEL_DIR --data_dir DATA_DIR --output_dir OUTPUT_DIR
transfer_input_files = baseline_inference.py,run_inference.sh,MODEL_DIR/,DATA_DIR/
output         = logs/job_$(Cluster)_$(Process).out
error          = logs/job_$(Cluster)_$(Process).err
log            = logs/job_$(Cluster)_$(Process).log
notification = complete
transfer_executable = false
request_memory = 4*1024
queue
