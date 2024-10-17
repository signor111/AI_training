import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_sentences(model_path, num_sentences=10, max_length=50):
    # 加载微调后的模型
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # 使用原始的GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 设置填充标记
    tokenizer.pad_token = tokenizer.eos_token

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置生成参数
    generate_params = {
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 1,
        "max_length": max_length,
    }

    # 生成句子
    sentences = []
    for _ in range(num_sentences):
        # 添加一个简短的提示作为起始点
        prompt = "Once upon a time"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # 检查输入是否为空
        if input_ids.nelement() == 0:
            print("Warning: Empty input. Skipping this generation.")
            continue

        try:
            output = model.generate(input_ids, **generate_params)
            sentence = tokenizer.decode(output[0], skip_special_tokens=True)
            sentences.append(sentence)
        except RuntimeError as e:
            print(f"Error during generation: {e}")
            continue

    return sentences


if __name__ == "__main__":
    # 使用函数生成句子
    model_path = ".\\fine_tuned_gpt2"
    generated_sentences = generate_sentences(model_path)

    # 打印生成的句子
    for i, sentence in enumerate(generated_sentences, 1):
        print(f"Sentence {i}: {sentence}")