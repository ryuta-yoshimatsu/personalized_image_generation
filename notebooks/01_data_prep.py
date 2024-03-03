# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at https://github.com/databricks-industry-solutions.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install requirements and load helper functions
# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md
# MAGIC #Prepare your images for fine-tuning
# MAGIC  Tailoring the output of a generative model is crucial for building a successful application. This applies to use cases powered by an image generation model as well. For example, a furniture designer wants to see their previous designs reflected on a newly generated image. But they also want to see some modifications, for example in material or color. In such case, it is important that the model is aware of their previous products and can apply new styles to generate new product designs. Customization is necessary in a case like this. We can do this by fine-tuning a pre-trained model on our own images.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manage your images in Unity Catalog Volumes

# COMMAND ----------

# MAGIC %md
# MAGIC This solution accelerator uses the 25 training images stored in the subfolders of ```/images/chair/``` to fine-tune a model. We copy the images to Unity Catalog (UC) and managed them as volume files. To adapt this solution to your use case, you can directly upload your images in UC volumes.

# COMMAND ----------

theme = "chair"
catalog = "sdxl_image_gen" # Name of the catalog we use to manage our assets (e.g. images, weights, datasets) 
volumes_dir = f"/Volumes/{catalog}/{theme}" # Path to the directories in UC Volumes

# COMMAND ----------

# MAGIC %sql CREATE CATALOG IF NOT EXISTS sdxl_image_gen

# COMMAND ----------

# MAGIC %sql CREATE SCHEMA IF NOT EXISTS sdxl_image_gen.chair

# COMMAND ----------

import os
import subprocess

# Create volumes under the schma, and copy the training images into it 
for volume in os.listdir("../images/chair"):
  volume_name = f"{catalog}.{theme}.{volume}"
  spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_name}")
  command = f"cp ../images/chair/{volume}/*.jpg /Volumes/{catalog}/{theme}/{volume}/"
  process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
  output, error = process.communicate()
  if error:
    print('Output: ', output)
    print('Error: ', error)

# COMMAND ----------

import glob

# Display images in Volumes
img_paths = f"{volumes_dir}/*/*.jpg"
imgs = [PIL.Image.open(path) for path in glob.glob(img_paths)]
num_imgs_to_preview = 25
show_image_grid(imgs[:num_imgs_to_preview], 5, 5) # Custom function defined in util notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Annotate your images with a unique token
# MAGIC Note that the 25 images above consists of 5 different styles of chair and each style has 5 images. We need to provide a caption for each of the images associated with a given style of chair. An important thing here is to provide a unique token for each style and use it in the captions: e.g. “A photo of a BCNCHR chair”, where BCNCHR is the unique token assigned to the black leather chair in top row. The uniqueness of the token helps us preserve the syntactic and semantic knowledge that the base pre-trained model brings by default. The idea of fine-tuning is not to mess up with what the model knows already, and to encode a new token and learn the association between that token and the subject. Read more about this [here](https://dreambooth.github.io/).
# MAGIC
# MAGIC We add a token (e.g. BCNCHR) to each caption using a caption prefix. For this example, we use "a photo of a BCNCHR chair," but other options include: "a photo of a chair in the style of BCNCHR".

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automate the generation of custom captions with BLIP
# MAGIC When we have too many training images, automating the caption generation using a model like BLIP is also an option. 

# COMMAND ----------

import pandas as pd
import PIL
import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

# load the processor and the captioning model
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large", torch_dtype=torch.float16
).to(device)

# COMMAND ----------

# create a list of (Pil.Image, path) pairs
imgs_and_paths = [
    (path, PIL.Image.open(path).rotate(-90))
    for path in glob.glob(f"{volumes_dir}/*/*.jpg")
]

# COMMAND ----------

import json

captions = []
for img in imgs_and_paths:
    instance_class = img[0].split("/")[4].replace("_", " ")
    caption_prefix = f"a photo of a {instance_class} {theme}: "
    caption = (
        caption_prefix
        + caption_images(img[1], blip_processor, blip_model, device).split("\n")[0] # Function caption_images is defined in utils notebook 
    )
    captions.append(caption)

# COMMAND ----------

# Show the captions generated by BLIP
display(pd.DataFrame(captions).rename(columns={0: "caption"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manage Dataset in UC Volumes
# MAGIC We create a Hugging Face Dataset object and store it in Unity Catalog Volume.

# COMMAND ----------

from datasets import Dataset, Image

d = {
    "image": [imgs[0] for imgs in imgs_and_paths],
    "caption": [caption for caption in captions],
}
dataset = Dataset.from_dict(d).cast_column("image", Image())
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{theme}.dataset")
dataset.save_to_disk(f"/Volumes/{catalog}/{theme}/dataset")

# COMMAND ----------

# MAGIC %md Let's free up some memory again.

# COMMAND ----------

import gc
del blip_processor, blip_model
gc.collect()
torch.cuda.empty_cache()
