GPU_ID=3,4,5

# For ALBERT-xxlarge, change learning_rate from 2e-5 to 1e-5
#
mkdir hwameiner_models


for seed in 32; do
CUDA_VISIBLE_DEVICES=$GPU_ID  python3 run_hwameiner_gongkai.py  --model_type bertspanmarker  \
    --model_name_or_path  ../data/bert_models/bert-base-hm-wwm-20e-384b-15m  --do_lower_case  \
    --data_dir ../DataSet/hwamei_gongkai  \
    --learning_rate 2e-5  --num_train_epochs 10  --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 16  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --save_steps 20000  --max_pair_length 256  --max_mention_ori_length 28    \
    --do_train --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed $seed  --onedropout  \
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_results \
    --output_dir ./hwameiner_models/gongkai_ner-all-hm-wwm-15m--$seed  --overwrite_output_dir  --output_results \
    --do_test \
    --add_cixinxi  \
    --add_binglixinxi2 3\
    --ci_use_diff \
    #--add_binglixinxi 1\
    #--continue_train_from_saved_model ./hwameiner_models/PL-Marker-hwamei-bert-40-3/checkpoint-103775  \
    #--add_binglixinxi2 3 \

    #--lminit
done;

