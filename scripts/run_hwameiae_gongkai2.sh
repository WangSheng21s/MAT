GPU_ID=3,4,5


for seed in  32 33 34 35 36;do
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_hwameiae_gongkai.py  --model_type bertsub  \
    --model_name_or_path   ./bert_models/bert-base-hm-wwm-20e-384b-15m --do_lower_case  \
    --data_dir ./DataSet/hwamei_500  \
    --learning_rate 2e-5  --num_train_epochs 100  --per_gpu_train_batch_size  16  --per_gpu_eval_batch_size 32  --gradient_accumulation_steps 1  \
    --max_seq_length 512  --max_pair_length 8  --save_steps 10000  \
    --fp16  --seed $seed      \
    --test_file ./hwameiner_models/gongkai_ner-all-hm-wwm-15m--$seed/ent_pred_test.json  \
    --use_ner_results \
    --output_dir ./hwameiae_models/gongkai-blxx-3-$sedd --overwrite_output_dir \
    --do_train --do_eval  --evaluate_during_training   --eval_all_checkpoints  --eval_logsoftmax  \
    --add_binglixinxi 3 \
    --dont_use_ner_loss \


done;

