# 训练
python /home/aistudio/Janus-main/train-lora.py --json_path /home/aistudio/Janus-main/open.json --image_dir /home/aistudio/Janus-main/image --pretrained_model /home/aistudio/JanusPro1B --output_dir /home/aistudio/Janus-main/output-lora --batch_size 8 --max_epochs 10
# 推理
# python /home/aistudio/Janus-main/app.py --model_path /home/aistudio/JanusPro1B --lora_path /home/aistudio/Janus-main/output-loratest 