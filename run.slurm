#!/bin/bash
#SBATCH --job-name=self-instruct
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --exclusive
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:8
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/armel/Self-instruct/logs/%x-%j.out
#SBATCH --mem-per-cpu=11G

set -x -e

source /admin/home/armel/.bashrc

conda activate finetune

echo "START TIME: $(date)"

# Training setup
GPUS_PER_NODE=8
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


export WANDB_PROJECT=test
export HF_DATASETS_CACHE="/fsx/armel/.cache"

PATH_TO_LOG=/fsx/armel/Self-instruct/logs/

LOG_PATH=$PATH_TO_LOG/test_log.txt

CMD=" \
    main.py \
    --seed_tasks_path="data/code_tasks.jsonl" \
    --output_data_path data/bi/code_llama_34b/output.jsonl \
    --num_instructions_to_generate 20000 \
    --template_name better \
    --format 2 \
    --model_name_or_path="codellama/CodeLlama-34b-hf" \
    --num_prompt_instructions 8 \
    --request_batch_size 8 \
    --num_prompt_synthetic_instructions 2 \
    --max_new_tokens 4096 \
    --temperature 0.8 \
    --top_p 0.95 \
    --num_beams 1 \
    --repetition_penalty 1.2 \
    --threshold 0.7 \
    --seed 42 \
    --keep_programming \
"
#    --use_tgi \

#CMD=" \
#    processing.py \
#    --seed_tasks_path="data/code_tasks.jsonl" \
#    --input_data_path data/output.jsonl \
#    --output_data_path data/output_processed.jsonl \
#    --template_name default \
#    --model_name_or_path="bigcode/starcoderbase-1b" \
#    --num_prompt_instructions 4 \
#    --num_trials 1 \
#    --max_new_tokens 512 \
#    --temperature 0.2 \
#    --top_p 0.95 \
#    --num_beams 1 \
#    --repetition_penalty 1.2 \
#    --threshold 0.7 \
#    --seed 42 \
#"

#CMD=" \
#    tgi.py
#"
#export LAUNCHER="accelerate launch \
#    --multi_gpu \
#    --num_machines $NNODES \
#    --num_processes $WORLD_SIZE \
#    --main_process_ip "$MASTER_ADDR" \
#    --main_process_port $MASTER_PORT \
#    --num_processes $WORLD_SIZE \
#    --machine_rank \$SLURM_PROCID \
#    --role $SLURMD_NODENAME: \
#    --rdzv_conf rdzv_backend=c10d \
#    --max_restarts 0 \
#    --tee 3 \
#    "

export LAUNCHER="python3 -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# AWS specific
export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens

export CUDA_HOME=/usr/local/cuda-11.6
export LD_PRELOAD=$CUDA_HOME/lib/libnccl.so
export LD_LIBRARY_PATH=$CUDA_HOME/efa/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# py-spy top -s -i -n -- $LAUNCHER --node_rank $SLURM_PROCID --role $SLURMD_NODENAME: $CMD
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD"

echo "END TIME: $(date)"
