
import argparse
import torch
import gradio as gr
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor
from janus.utils.io import load_pil_images
from peft import PeftModel

# 全局变量用于缓存加载的模型
MODEL = None
TOKENIZER = None
PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models(model_path, lora_path=None):
    """加载模型并缓存到全局变量"""
    global MODEL, TOKENIZER, PROCESSOR
    
    if MODEL is None:
        print(f"[INFO] Loading VLChatProcessor from {model_path}...")
        PROCESSOR = VLChatProcessor.from_pretrained(model_path)
        TOKENIZER = PROCESSOR.tokenizer

        print(f"[INFO] Loading base model from {model_path}...")
        base_model = (
            AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            .to(torch.bfloat16)
            .to(DEVICE)
            .eval()
        )

        if lora_path:
            print(f"[INFO] Loading LoRA adapter from {lora_path}...")
            MODEL = PeftModel.from_pretrained(base_model, lora_path)
        else:
            MODEL = base_model

        MODEL.eval()

def analyze_image(image, question, max_new_tokens=512):
    """处理图像和问题并生成回答"""
    if image is None:
        return "请先上传一张图片"
    
    # 创建对话结构
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    try:
        # 处理输入
        pil_images = load_pil_images(conversation)
        prepare_inputs = PROCESSOR(
            conversations=conversation, 
            images=pil_images, 
            force_batchify=True
        ).to(DEVICE)

        # 生成回答
        inputs_embeds = MODEL.prepare_inputs_embeds(**prepare_inputs)
        outputs = MODEL.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=TOKENIZER.eos_token_id,
            bos_token_id=TOKENIZER.bos_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        answer = TOKENIZER.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"处理过程中发生错误: {str(e)}"

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,default="/home/aistudio/JanusPro1B")
    parser.add_argument("--lora_path", type=str,default="/home/aistudio/Janus-main/output-loratest")
    args = parser.parse_args()

    # 加载模型
    load_models(args.model_path, args.lora_path)

    # 创建Gradio界面
    with gr.Blocks(css="""
        .title {
            text-align: center;
            color: #1a3d6d;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 15px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .subtitle {
            text-align: center;
            color: #444;
            font-size: 18px;
            margin-bottom: 35px;
            font-style: italic;
        }
        .input-container {
            background-color: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        }
        .output-container {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        }
        .image-upload {
            border: 2px dashed #aab7c4;
            border-radius: 10px;
            padding: 25px;
            text-align: center;
            margin-bottom: 25px;
            background-color: #fdfdff;
            transition: border-color 0.3s ease;
        }
        .image-upload:hover {
            border-color: #27ae60;
        }
        .image-upload h3 {
            color: #333;
            margin-top: 12px;
            font-size: 16px;
        }
        .question-box {
            margin-bottom: 20px;
        }
        .control-row {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            align-items: center;
        }
        .btn-primary {
            background-color: #27ae60;
            color: blue;
            border: none;
            border-radius: 8px;
            padding: 14px 28px;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(26, 115, 232, 1);
        }
        .btn-primary:hover {
            background-color: #219150;
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(13, 71, 161, 1);
        }
        .btn-primary:active {
            transform: translateY(0);
        }
        .slider-container {
            flex: 2;
        }
        .slider-label {
            display: block;
            margin-bottom: 8px;
            font-size: 15px;
            color: #333;
            font-weight: 500;
        }
        .output-text {
            font-size: 17px;
            line-height: 1.8;
            color: #2c3e50;
            white-space: pre-wrap;
            min-height: 180px;
            padding: 15px;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .loading {
            text-align: center;
            margin: 25px 0;
            color: #27ae60;
            font-weight: bold;
        }
        .error-message {
            color: #e67e22;
            text-align: center;
            margin-top: 12px;
            font-size: 15px;
        }
        @media (max-width: 768px) {
            .control-row {
                flex-direction: column;
            }
            .slider-container {
                width: 100%;
            }
        }
    """) as demo:
        gr.Markdown('<div class="title">胸部X光智能分析系统</div>')
        gr.Markdown('<div class="subtitle">上传胸部X光片，获取智能分析</div>')
        
        with gr.Row():
            # 左侧输入区域
            with gr.Column(scale=1, elem_classes=["input-container"]):
                image_input = gr.Image(
                    label="上传胸部X光片", 
                    type="filepath", 
                    elem_classes=["image-upload"],
                    height=300
                )
                question_input = gr.Textbox(
                    label="输入问题", 
                    value="通过这张胸部x光可以诊断出什么?",  # 设置为中文默认问题
                    elem_classes=["question-box"],
                    lines=2
                )
                
                with gr.Row(elem_classes=["control-row"]):
                    max_tokens = gr.Slider(
                        32, 1024, 
                        value=512, 
                        label="最大生成长度",
                        elem_classes=["slider-container"]
                    )
                    submit_btn = gr.Button("🔍 开始分析", elem_classes=["btn-primary"])
            
            # 右侧输出区域
            with gr.Column(scale=1, elem_classes=["output-container"]):
                output_text = gr.Textbox(
                    label="智能分析结果", 
                    interactive=False, 
                    elem_classes=["output-text"],
                    lines=8
                )
                loading = gr.Markdown(
                    '<div class="loading">模型已加载，可以开始使用</div>', 
                    visible=True
                )
                gr.Markdown('<div class="footer">© Euan(2025) 胸部X光智能分析系统.</div>')

        # 绑定事件
        submit_btn.click(
            fn=analyze_image,
            inputs=[image_input, question_input, max_tokens],
            outputs=output_text
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()