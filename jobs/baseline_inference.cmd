executable = baseline_inference.sh
getenv     = true
arguments  = 
transfer_input_files = baseline_run.sh, baseline_inference.py
output     = baseline_run.out
error      = baseline_run.error
log        = baseline_run.log
notification = complete
transfer_executable = false
request_memory = 4*1024
queue