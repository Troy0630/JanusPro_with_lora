train-lora.py：lora微调janus。

app.py:使用模型的推理界面。

![structure](structure.png)

# 用法：

## 训练：

python **/home/aistudio/Janus-main/train-lora.py** **--json_path** /home/aistudio/Janus-main/open.json **--image_dir** /home/aistudio/fine/image --pretrained_model /home/aistudio/JanusPro1B --output_dir /home/aistudio/Janus-main/output-lora --batch_size 8 --max_epochs 3

--**json_path** 为json路径

**--image_dir** 为image路径

**--pretrained_model** 为下载的Januspro模型权重，需自行下载

## 推理：

python /home/aistudio/Janus-main/app.py --model_path /home/aistudio/JanusPro1B --lora_path1 /home/aistudio/Janus-main/output-loratest --lora_path2 /home/aistudio/Janus-main/output-lorafine