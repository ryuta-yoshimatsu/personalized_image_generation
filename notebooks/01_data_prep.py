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
# MAGIC ## Use default images

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

# Make sure this directory exists. If not run: %sh mkdir /dbfs/tmp/sdxl/default_images/
local_dir_default = "/dbfs/tmp/sdxl/default_images/"

snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir_default, 
    repo_type="dataset",
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
img_paths = f"{local_dir_default}*.jpeg"
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

# create a list of (Pil.Image, path) pairs
imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{local_dir_default}*.jpeg")]

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

captions = []
caption_prefix = "a photo of corgi dog, " #@param
for img in imgs_and_paths:
    caption = caption_prefix + caption_images(img[1]).split("\n")[0]
    captions.append(caption)

# COMMAND ----------

from datasets import Dataset, Image
d = {
    "image": [imgs[0] for imgs in imgs_and_paths],
    "caption": [caption for caption in captions],
}

dataset = Dataset.from_dict(d).cast_column("image", Image())
dataset.save_to_disk('/dbfs/tmp/sdxl/default_data')

# COMMAND ----------

# MAGIC %md
# MAGIC ##Use your own images

# COMMAND ----------

local_dir = "/dbfs/tmp/sdxl/images/"

# COMMAND ----------

from PIL import Image

# change path to display images from your local dir
img_paths = f"{local_dir}*/*.jpg"
imgs = [Image.open(path) for path in glob.glob(img_paths)]

num_imgs_to_preview = 25
image_grid(imgs[:num_imgs_to_preview], 5, 5)

# COMMAND ----------

import glob
from PIL import Image

# create a list of (Pil.Image, path) pairs
imgs_and_paths = [(path,Image.open(path)) for path in glob.glob(f"{local_dir}*/*.jpg")]

# COMMAND ----------

import json

captions = []
for img in imgs_and_paths:
    instance_class = img[0].split("/")[5].replace("_", " ")
    caption_prefix = f"a photo of {instance_class} cat: "
    caption = caption_prefix + caption_images(img[1]).split("\n")[0]
    captions.append(caption)

# COMMAND ----------

from datasets import Dataset, Image
d = {
    "image": [imgs[0] for imgs in imgs_and_paths],
    "caption": [caption for caption in captions],
}

dataset = Dataset.from_dict(d).cast_column("image", Image())
dataset.save_to_disk('/dbfs/tmp/sdxl/data')

# COMMAND ----------

# MAGIC %md
# MAGIC Free some memory:

# COMMAND ----------

import gc

# delete the BLIP pipelines and free up some memory
del blip_processor, blip_model
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------


