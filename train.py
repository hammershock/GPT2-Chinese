"""
Train a Chinese gpt-model

input:
1. corpus lines
    a txt file, each line is a document sentence
2. vocabulary(Optional)
    a txt file, each line is a word
pretrained-model(Optional)

output:
trained model
training logs

"""
import os

import torch
from torch.utils.data import DataLoader
from tokenizer import make_tensor_dataset
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        labels=inputs['input_ids'])
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    attention_mask = torch.stack([item[1] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}


if __name__ == '__main__':
    data_root_dir = './RA2Parrot/datasets_/红警HBK08/subtitles'
    model_name = 'gpt2-distil-chinese-cluecorpussmall'
    checkpoint_dir = './results'

    dataset = make_tensor_dataset("./data/data.txt", "./data/vocabulary.txt")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GPT2LMHeadModel.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=64,
        save_steps=200,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir='./logs',
        logging_steps=500,
        report_to="tensorboard",
        save_strategy="steps",
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    if len(os.listdir(checkpoint_dir)):
        latest_checkpoint = max(os.listdir(checkpoint_dir), key=lambda x: int(x.split('.')[0].lstrip('checkpoint-')))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        trainer.train(resume_from_checkpoint=checkpoint_path if os.path.isdir(checkpoint_path) else None)
    else:
        trainer.train()

