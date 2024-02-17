# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at https://github.com/databricks-industry-solutions.

# COMMAND ----------

# MAGIC %run ./util

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate custom captions with BLIP
# MAGIC Load BLIP to auto caption images:

# COMMAND ----------

# load the processor and the captioning model
device = "cuda" if torch.cuda.is_available() else "cpu"

blip_processor = AutoProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large")

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large", 
    torch_dtype=torch.float16).to(device)

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

# MAGIC %md
# MAGIC ##Use your own images

# COMMAND ----------

# MAGIC %md Make sure you have uploaded your training images in the Volumes.

# COMMAND ----------

theme = "chair"
local_dir = f"/Volumes/sdxl/{theme}"

# COMMAND ----------

# change path to display images from your local dir
img_paths = f"{local_dir}/*/*.jpg"
imgs = [PIL.Image.open(path) for path in glob.glob(img_paths)]
num_imgs_to_preview = 25

show_image_grid(imgs[:num_imgs_to_preview], 5, 5)

# COMMAND ----------

# create a list of (Pil.Image, path) pairs
imgs_and_paths = [(path, PIL.Image.open(path).rotate(-90)) for path in glob.glob(f"{local_dir}/*/*.jpg")]

# COMMAND ----------

import json
captions = []
for img in imgs_and_paths:
    instance_class = img[0].split("/")[4].replace("_", " ")
    caption_prefix = f"a photo of {instance_class} {theme}: "
    caption = caption_prefix + caption_images(img[1], blip_processor, blip_model, device).split("\n")[0]
    captions.append(caption)

# COMMAND ----------

display(pd.DataFrame(captions).rename(columns={0: "caption"}))

# COMMAND ----------

from datasets import Dataset, Image
d = {
    "image": [imgs[0] for imgs in imgs_and_paths],
    "caption": [caption for caption in captions],
}
dataset = Dataset.from_dict(d).cast_column("image", Image())
dataset.save_to_disk(f'/Volumes/sdxl/{theme}/dataset')

# COMMAND ----------


