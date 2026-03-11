import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import bitsandbytes as bnb



"""
3. Load a Pretrained Quantized Model
Let's loads a 1.3B parameter model with 4-bit quantization to save memory. The device_map="auto" argument automatically assigns the model to the available GPU.
"""
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Enables 4-bit quantization
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Fine-tuning specific layers
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()



dataset = load_dataset("imdb", split="train[:10000]")  # Sentiment analysis dataset

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=3,
    fp16=True,  # Enable mixed precision training
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()