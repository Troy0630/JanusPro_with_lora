# 目录

.
|-- src
|   |`-- Janus
| |-- train.py
| |-- app.py
|-- scripts
|   `   |-- run.sh
|-- model.png
|-- dataset
|   |-- OpenI-ZH.json
| -- readme.md

train-lora.py：lora微调Janus-Pro；

Open-ZH.json: 数据集标签；

app.py:使用模型的推理界面。

# 模型

lora微调Janus-Pro模型架构图如下：

![model](model.png)

# 快速开始：

```bash
bash run.sh
```

## 训练：

```python
python /home/aistudio/Janus-main/train-lora.py --json_path/home/aistudio/Janus-main/open.json --image_dir /home/aistudio/Janus-main/image --pretrained_model /home/aistudio/JanusPro1B --output_dir /home/aistudio/Janus-main/output-lora --batch_size 8 --max_epochs 10
```

--**json_path** 为json路径

**--image_dir** 为image路径

**--pretrained_model** 为下载的Januspro模型权重，需自行下载

## 推理：

```python
python /home/aistudio/Janus-main/app.py --model_path /home/aistudio/JanusPro1B --lora_path /home/aistudio/Janus-main/output-loratest 
```

