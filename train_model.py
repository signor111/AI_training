from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载预训练模型和分词器
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 设置填充标记
    tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # 定义数据预处理函数
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    # 对数据集进行预处理
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])

    # 准备训练集和验证集
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    # 设置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_steps=100,  # 每100步记录一次日志
        evaluation_strategy="steps",  # 定期进行评估
        eval_steps=500,  # 每500步评估一次
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model("./fine_tuned_gpt2")

    print("Training completed!")