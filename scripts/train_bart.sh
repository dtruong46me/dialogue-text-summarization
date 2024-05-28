pip install -q -U datasets
pip install -q -r "requirements.txt"

echo "Complete setup!"

python "/kaggle/working/dialogue-text-summarization/run_training.py"\
--huggingface_hub_token "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"\
--wandb_token "c74fcec22fbb4be075a981b1f3db3f464b15b089"\
--checkpoint "facebook/bart-base"\
--datapath "knkarthick/dialogsum"\
--output_dir "train-bart-base"\
--overwrite_output_dir True\
--num_train_epochs 10\
--per_device_train_batch_size 4\
--per_device_eval_batch_size 4\
--gradient_accumulation_steps 2\
--learning_rate 0.00005\
--evaluation_strategy "epoch"\
--save_strategy "epoch"\
--logging_strategy "epoch"\
--save_total_limit 1\
--report_to "wandb"\
--run_name "train_bart"\
--min_new_tokens 10\
--max_new_tokens 256\
--temperature 0.8\
--top_p 1.0\
--top_k 50