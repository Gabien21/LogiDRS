#!/bin/bash
export RECLOR_DIR=logiqa_data
export TASK_NAME=logiqa
export MODEL_DIR=roberta-large 
export MODEL_TYPE=LogiDRS
export GRAPH_VERSION=4
export DATA_PROCESSING_VERSION=32
export MODEL_VERSION=2132
export GNN_VERSION=GCN
export SAVE_DIR=dagn

nvidia-smi #check cuda/gpu

python run_multiple_choice.py \
     --task_name $TASK_NAME \
     --model_type $MODEL_TYPE \
     --model_name_or_path $MODEL_DIR \
     --do_predict \
     --data_dir $RECLOR_DIR \
     --graph_building_block_version $GRAPH_VERSION \
     --data_processing_version $DATA_PROCESSING_VERSION \
     --use_discourse \
     --n_transformer_layer 1\
     --max_seq_length 256 \
     --per_device_eval_batch_size 8 \
     --per_device_train_batch_size 2 \
     --gradient_accumulation_steps 2 \
     --roberta_lr 7e-6 \
     --discourse_lr 7e-6 \
     --transformer_lr 7e-6 \
     --num_train_epochs 20 \
     --output_dir Checkpoints/$TASK_NAME/${SAVE_DIR} \
     --logging_strategy epoch \
     --evaluation_strategy epoch \
     --save_strategy epoch \
     --save_total_limit 1 \
     --adam_epsilon 1e-6 \
     --weight_decay 0.01 \
     --numnet_drop 0.2 \
     --load_best_model_at_end \
     --metric_for_best_model eval_acc