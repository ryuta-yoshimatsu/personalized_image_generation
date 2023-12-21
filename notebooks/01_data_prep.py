# Databricks notebook source
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Install dependencies.
%pip install bitsandbytes transformers accelerate --quiet
%pip install huggingface_hub --upgrade --quiet
%pip install git+https://github.com/microsoft/DeepSpeed --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC **Let's get our training data!**
# MAGIC For this example, we'll download some images from the hub
# MAGIC
# MAGIC If you already have a dataset on the hub you wish to use, you can skip this part and go straight to: "Prep for
# MAGIC training 💻" section, where you'll simply specify the dataset name.
# MAGIC
# MAGIC If your images are saved locally, and/or you want to add BLIP generated captions,
# MAGIC pick option 1 or 2 below.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Download example images from the hub:

# COMMAND ----------

from huggingface_hub import snapshot_download

local_dir = "/dbfs/tmp/ryuta/sdxl/dog/"

snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)

# COMMAND ----------

# MAGIC %md
# MAGIC Preview the images:

# COMMAND ----------

from PIL import Image

def image_grid(imgs, rows, cols, resize=256):

    if resize is not None:
        imgs = [img.resize((resize, resize)) for img in imgs]
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# COMMAND ----------

import glob

# change path to display images from your local dir
img_paths = f"{local_dir}*.jpeg"
imgs = [Image.open(path) for path in glob.glob(img_paths)]

num_imgs_to_preview = 5
image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate custom captions with BLIP
# MAGIC Load BLIP to auto caption your images:

# COMMAND ----------

import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the processor and the captioning model
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",torch_dtype=torch.float16).to(device)

# captioning utility
def caption_images(input_image):
    inputs = blip_processor(images=input_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs.pixel_values

    generated_ids = blip_model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# COMMAND ----------

import glob
from PIL import Image

# create a list of (Pil.Image, path) pairs
imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{local_dir}*.jpeg")]

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's add the concept token identifier (e.g. TOK) to each caption using a caption prefix.
# MAGIC Feel free to change the prefix according to the concept you're training on!
# MAGIC - for this example we can use "a photo of TOK," other options include:
# MAGIC     - For styles - "In the style of TOK"
# MAGIC     - For faces - "photo of a TOK person"
# MAGIC - You can add additional identifiers to the prefix that can help steer the model in the right direction.
# MAGIC -- e.g. for this example, instead of "a photo of TOK" we can use "a photo of TOK dog" / "a photo of TOK corgi dog"

# COMMAND ----------

import json

caption_prefix = "a photo of TOK dog, " #@param
with open(f'{local_dir}metadata.jsonl', 'w') as outfile:
  for img in imgs_and_paths:
      caption = caption_prefix + caption_images(img[1]).split("\n")[0]
      entry = {"file_name":img[0].split("/")[-1], "prompt": caption}
      json.dump(entry, outfile)
      outfile.write('\n')

# COMMAND ----------

dbutils.fs.head("dbfs:/tmp/ryuta/sdxl/dog/metadata.jsonl")

# COMMAND ----------

# MAGIC %md
# MAGIC Free some memory:

# COMMAND ----------

import gc

# delete the BLIP pipelines and free up some memory
del blip_processor, blip_model
gc.collect()
torch.cuda.empty_cache()
