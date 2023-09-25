#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import VisionEncoderDecoderModel,GPT2TokenizerFast,ViTImageProcessor,Seq2SeqTrainer,Seq2SeqTrainingArguments
from datasets import load_dataset
import numpy as np
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from peft import get_peft_model, LoraConfig,TaskType
from textwrap import wrap
from evaluate import load


# In[2]:


encoder_model = 'microsoft/swin-base-patch4-window7-224-in22k'
decoder_model ='gpt2'


# In[3]:


model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_model,
    decoder_model).to(device)


# In[4]:


tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
image_processor =ViTImageProcessor.from_pretrained(encoder_model)


# In[5]:


if "gpt2" in decoder_model:
  # gpt2 does not have decoder_start_token_id and pad_token_id
  # but has bos_token_id and eos_token_id
  tokenizer.pad_token = tokenizer.eos_token # pad_token_id as eos_token_id
  model.config.eos_token_id = tokenizer.eos_token_id
  model.config.pad_token_id = tokenizer.pad_token_id
  # set decoder_start_token_id as bos_token_id
  model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
  # set the decoder start token id to the CLS token id of the tokenizer
  model.config.decoder_start_token_id = tokenizer.cls_token_id
  # set the pad token id to the pad token id of the tokenizer
  model.config.pad_token_id = tokenizer.pad_token_id



# peft_config = LoraConfig(task_type='image_caption',
#                          target_modules = ["query", "value"],
#                          inference_mode=False, 
#                          r=8, 
#                          lora_alpha=32, 
#                          lora_dropout=0.1)

# model = get_peft_model(model,peft_config)


# model.print_trainable_parameters()




root='/mnt/storage-ssd/wangcheng/dataset/rgb/GIT/total/'

ds = load_dataset(root)


# In[10]:


ds = ds['train'].train_test_split(0.1)
test_ds = ds['test']


# In[11]:


ds = ds['train'].train_test_split(0.15)
train_ds = ds['train']
valid_ds = ds['test']


# In[12]:


len(train_ds),len(valid_ds),len(test_ds)


# In[13]:


max_length = 16 # max length of the captions in tokens


def preprocess(items):
  # preprocess the image
  pixel_values = image_processor(items["image"], return_tensors="pt").pixel_values.to(device)
  # tokenize the caption with truncation and padding
  targets = tokenizer(items['text'], 
                      max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
#   inputs = image_processor(images=pixel_values, text=targets, padding="max_length")
#   inputs.update({"labels": inputs["input_ids"]})
#   return inputs
  return {'pixel_values': pixel_values, 'labels': targets["input_ids"]}

train_dataset = train_ds.with_transform(preprocess)
valid_dataset = valid_ds.with_transform(preprocess)
test_dataset  = test_ds.with_transform(preprocess)


# In[16]:


# a function we'll use to collate the batches
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }


# In[17]:


import evaluate
# load the rouge and bleu metrics
rouge = evaluate.load("rouge")


# In[18]:


from bleu_script.bleu import Bleu
bleu =Bleu()


# In[19]:


def compute_metrics(eval_pred):
  preds = eval_pred.label_ids
  labels = eval_pred.predictions
  # decode the predictions and labels
  pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
  labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
  # compute the rouge score
  rouge_result = rouge.compute(predictions=pred_str, references=labels_str)
  # multiply by 100 to get the same scale as the rouge score
  rouge_result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
  # compute the bleu score
  bleu_result = bleu.compute(predictions=pred_str, references=labels_str)
  # get the length of the generated captions
  generation_length = bleu_result["translation_length"]
  return {
        **rouge_result, 
        "bleu": round(bleu_result["bleu"] * 100, 4), 
        "gen_len": bleu_result["translation_length"] / len(preds)
  }


# In[20]:


num_epochs = 50 # number of epochs
batch_size = 64 # the size of batches


# In[22]:


# define the training arguments
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,             # use generate to calculate the loss
    num_train_epochs=num_epochs,            # number of epochs
    evaluation_strategy="steps",            # evaluate after each eval_steps
    eval_steps=250,                        # evaluate after each 500 steps
    logging_steps=250,                     # log after each 500 steps
    save_steps=250,                       # save after each 500 steps
    per_device_train_batch_size=batch_size, # batch size for training
    per_device_eval_batch_size=batch_size,  # batch size for evaluation
    output_dir="vit-swin-base-224-gpt2-galaxy-captioning", # output directory
    # push_to_hub=True # whether you want to push the model to the hub,
    # check this guide for more details: https://huggingface.co/transformers/model_sharing.html
)


# In[23]:


# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,                     # the instantiated 🤗 Transformers model to be trained
    tokenizer=image_processor,       # we use the image processor as the tokenizer
    args=training_args,              # pass the training arguments
    compute_metrics=compute_metrics, 
    train_dataset=train_dataset,     
    eval_dataset=valid_dataset,      
    data_collator=collate_fn,        
)


# In[24]:


from torch.utils.data import DataLoader

def get_eval_loader(eval_dataset=None):
  return DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=batch_size)

def get_test_loader(eval_dataset=None):
  return DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size)

# override the get_train_dataloader, get_eval_dataloader and
# get_test_dataloader methods of the trainer
# so that we can properly load the data
trainer.get_train_dataloader = lambda: DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size)
trainer.get_eval_dataloader = get_eval_loader
trainer.get_test_dataloader = get_test_loader


# In[25]:


# train the model
trainer.train()





