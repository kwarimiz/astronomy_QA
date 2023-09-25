#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import requests
from PIL import Image
import transformers
from transformers import Blip2Processor ,Blip2Model,TrainingArguments,Trainer,BlipModel
import accelerate
from peft import get_peft_model, LoraConfig,TaskType
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
from evaluate import load


# In[2]:


model_checkpoint = 'Salesforce/blip-image-captioning-base'


# In[3]:


processor = Blip2Processor.from_pretrained(model_checkpoint)


# In[4]:


model = BlipModel.from_pretrained(model_checkpoint)


# In[5]:


peft_config = LoraConfig(task_type='image_caption',
                         target_modules = ["q", "v"],
                         inference_mode=False, 
                         r=8, 
                         lora_alpha=32, 
                         lora_dropout=0.1)


# In[6]:


model = get_peft_model(model,peft_config)


# In[7]:


model.print_trainable_parameters()


# In[8]:


root='/mnt/storage-ssd/wangcheng/dataset/rgb/GIT/total_resize/'


# In[9]:


ds = load_dataset(root)


ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]



def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions)
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


train_ds.set_transform(transforms)
test_ds.set_transform(transforms)


# In[16]:


wer = load("wer")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}


# In[17]:


model_name = model_checkpoint.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-galaxy",
    learning_rate=5e-5,
    num_train_epochs=50,
    fp16=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=50,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
    load_best_model_at_end=True,
)


# In[18]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)


# In[ ]:


trainer.train()

