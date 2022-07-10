#!/usr/bin/python3

import datasets
from transformers import DataCollatorWithPadding
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import TrainingArguments, Trainer

# load dataset
dataset = datasets.load_dataset(
    'parquet', 
    data_files={
        'train': 'data/train_UPD.parquet',
        'test': 'data/test_UPD.parquet',
    },
)

# load model and tokenizer
model_path = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=3)

# set PAD token to EOS 
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# setting id2label and vice-versa in config
id2label = {
    0: 'Ineffective',
    1: 'Adequate',
    2: 'Effective',
}
label2id = dict((v, k) for k, v in id2label.items())

model.config.id2label = id2label
model.config.label2id = label2id

# preprocess function to tokenize examples
def preprocess_function(examples):
    return tokenizer(examples['discourse_text_UPD'], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# collator to dynamically pad examples to longest example and convert to 
# Tensor
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model('model/gpt2-argument-effectiveness-classifier')
