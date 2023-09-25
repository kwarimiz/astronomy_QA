#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset

from torch.utils.data import Dataset,DataLoader

from transformers import AutoProcessor,AutoModelForCausalLM,TrainingArguments,Trainer

from textwrap import wrap
from evaluate import load
import torch
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


root='your data path'


# In[3]:


ds = load_dataset(root)


# In[4]:


ds


# In[5]:


ds = ds["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
test_ds = ds["test"]


# In[6]:


def plot_images(images,captions):
    plt.figure(figsize=(20,20))
    for i in range(len(images)):
        ax = plt.subplot(1,len(images),i+1)
        caption = captions[i]
        caption = '\n'.join(wrap(caption,12))
        plt.title(caption)
        plt.imshow(images[i])
        plt.axis('off')


# In[7]:


sample_images_to_visualize = [np.array(train_ds[i]["image"]) for i in range(5)]
sample_captions = [train_ds[i]["text"] for i in range(5)]
plot_images(sample_images_to_visualize, sample_captions)


# 

# In[8]:


checkpoint ='microsoft/git-base'
processor = AutoProcessor.from_pretrained(checkpoint)


# In[9]:


def transforms(example_batch):
    images = [x for x in example_batch["image"]]
    captions = [x for x in example_batch["text"]]
    inputs = processor(images=images, text=captions, padding="max_length")
    inputs.update({"labels": inputs["input_ids"]})
    return inputs


train_ds.set_transform(transforms)
test_ds.set_transform(transforms)


# In[10]:


model = AutoModelForCausalLM.from_pretrained(checkpoint)


# In[11]:


wer = load("wer")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predicted = logits.argmax(-1)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions = processor.batch_decode(predicted, skip_special_tokens=True)
    wer_score = wer.compute(predictions=decoded_predictions, references=decoded_labels)
    return {"wer_score": wer_score}


# In[12]:


model_name = checkpoint.split("/")[1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-galaxy",
    learning_rate=5e-5,
    num_train_epochs=50,
    fp16=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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


# In[13]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)


# In[14]:


trainer.train()


# In[ ]:




