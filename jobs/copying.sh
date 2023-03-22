#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bm4g15@soton.ac.uk
#SBATCH --output /scratch/bm4g15/data/rims/logs/copying/slurm/slurm-%A_%a.out       # todo: Make sure folder path exists and matches exp name. (Same for err dir).
#SBATCH --error /scratch/bm4g15/data/rims/logs/copying/slurm/errors/slurm-%A_%a.err

# Run command: sbatch copying.sh <KEY LOOKUP> <ID> <SEED>

module load conda/py3-latest
source deactivate
conda activate rim-env
cd /home/bm4g15/rim-modularisation/

export PYTHONPATH=./
export WANDB_MODE=offline

mkdir -p /scratch/bm4g15/data/rims/logs/copying/slurm/errors/
id=$2
seed=$3
num_gpus=1
num_workers=4
log_path=/scratch/bm4g15/data/rims/logs/
name_prefix='copying'

# https://linuxize.com/post/bash-case-statement/
case $1 in

  -1)
    name_postfix='test'
    python3 experiments/copying.py \
      --num_gpus 0 --id -1 --wandb_notes "sanity check iridis setup (10 min timeout)" \
      --cell VanillaLSTMCell --use_vanilla_lstm_layer --num_rims_per_layer 1 --num_active_rims_per_layer 1 \
      --d_enc_inp 300 --d_hid_all_rims_per_layer 300 \
      --log_last_n 10 --num_workers ${num_workers} \
      --log_path ${log_path} --name_prefix ${name_prefix} --name_postfix ${name_postfix}
    ;;

  VanillaLSTMCell_h300)
    name_postfix='h300'
    python3 experiments/copying.py \
      --id ${id} --seed ${seed} \
      --cell VanillaLSTMCell --use_vanilla_lstm_layer --num_rims_per_layer 1 --num_active_rims_per_layer 1 \
      --d_hid_all_rims_per_layer 300 --d_enc_inp 300 \
      --log_last_n 10 --num_workers ${num_workers} \
      --log_path ${log_path} --name_prefix ${name_prefix} --name_postfix ${name_postfix}
    ;;

  VanillaLSTMCell_h600)
    name_postfix='h600'
    python3 experiments/copying.py \
      --id ${id} --seed ${seed} \
      --cell VanillaLSTMCell --use_vanilla_lstm_layer --num_rims_per_layer 1 --num_active_rims_per_layer 1 \
      --d_hid_all_rims_per_layer 600 --d_enc_inp 600 \
      --log_last_n 10 --num_workers ${num_workers} \
      --log_path ${log_path} --name_prefix ${name_prefix} --name_postfix ${name_postfix}
    ;;

  BlockCellLSTM_h600)
    name_postfix='h600'
    python3 experiments/copying.py \
      --id ${id} --seed ${seed} --comm_residual --wandb_notes "comm_residual=True" \
      --cell BlockCellLSTM --use_comm_attn \
      --d_hid_all_rims_per_layer 600 --num_rims_per_layer 6 \
      --log_last_n 10 --num_workers ${num_workers} \
      --log_path ${log_path} --name_prefix ${name_prefix} --name_postfix ${name_postfix}
    ;;

  BatchCellLSTM_h600)
    name_postfix='h600'
    python3 experiments/copying.py \
      --id ${id} --seed ${seed} --comm_residual --wandb_notes "comm_residual=True" \
      --cell BatchCellLSTM --use_comm_attn \
      --d_hid_all_rims_per_layer 600 --num_rims_per_layer 6 \
      --log_last_n 10 --num_workers ${num_workers} --num_gpus ${num_gpus} \
      --log_path ${log_path} --name_prefix ${name_prefix} --name_postfix ${name_postfix}
    ;;

  ######################################################################################################################
  # Sanity checks
  VanillaLSTMCell_h600_clip0)
    name_postfix='h600_clip0'
    python3 experiments/copying.py \
      --id ${id} --seed ${seed} \
      --grad_clip_value 0 \
      --cell VanillaLSTMCell --use_vanilla_lstm_layer --num_rims_per_layer 1 --num_active_rims_per_layer 1 \
      --d_hid_all_rims_per_layer 600 \
      --log_last_n 10 --num_workers ${num_workers} \
      --log_path ${log_path} --name_prefix ${name_prefix} --name_postfix ${name_postfix}
    ;;

  VanillaLSTMCell_h600_drop0)
    name_postfix='h600_drop0'
    python3 experiments/copying.py \
      --id ${id} --seed ${seed} \
      --dropout 0 \
      --cell VanillaLSTMCell --use_vanilla_lstm_layer --num_rims_per_layer 1 --num_active_rims_per_layer 1 \
      --d_hid_all_rims_per_layer 600 \
      --log_last_n 10 --num_workers ${num_workers} \
      --log_path ${log_path} --name_prefix ${name_prefix} --name_postfix ${name_postfix}
    ;;

  VanillaLSTMCell_h600_clip0_drop0)
    name_postfix='h600_clip0_drop0'
    python3 experiments/copying.py \
      --id ${id} --seed ${seed} \
      --grad_clip_value 0 --dropout 0 \
      --cell VanillaLSTMCell --use_vanilla_lstm_layer --num_rims_per_layer 1 --num_active_rims_per_layer 1 \
      --d_hid_all_rims_per_layer 600 \
      --log_last_n 10 --num_workers ${num_workers} \
      --log_path ${log_path} --name_prefix ${name_prefix} --name_postfix ${name_postfix}
    ;;

  ######################################################################################################################
  # ABLATIONs
  # RIMs paper - see table 4 (appendix) - 100 epochs
  BlockCellLSTM_h600_iAtT-cAtF_kT4-kA2)
    name_postfix='h600_iAtT-cAtF_kT4-kA2'
    python3 experiments/copying.py \
      --id ${id} --seed ${seed} \
      --cell BlockCellLSTM --max_epochs 100 \
      --d_hid_all_rims_per_layer 600 --num_rims_per_layer 4 --num_active_rims_per_layer 2 \
      --log_last_n 10 --num_workers ${num_workers} \
      --log_path ${log_path} --name_prefix ${name_prefix} --name_postfix ${name_postfix}
    ;;

  *)
    echo -n "INVALID EXPERIMENT ID GIVEN. SWITCH CASE FAILED!"
    ;;
esac










