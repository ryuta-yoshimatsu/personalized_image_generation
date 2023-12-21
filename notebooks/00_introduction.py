# Databricks notebook source
# MAGIC %md
# MAGIC ##Base

# COMMAND ----------

# MAGIC %pip install diffusers --upgrade --quiet
# MAGIC %pip install transformers accelerate safetensors  --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC To just use the base model, you can run:

# COMMAND ----------

from diffusers import DiffusionPipeline
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = DiffusionPipeline.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0", 
  torch_dtype=torch.float16, 
  use_safetensors=True, 
  variant="fp16"
)
pipe.to(device)

# COMMAND ----------

prompt = "A photo of a dog eating dog food"
images = pipe(prompt=prompt).images[0]

# COMMAND ----------

images

# COMMAND ----------

# MAGIC %md
# MAGIC ###Base + Refiner

# COMMAND ----------

# MAGIC %pip install diffusers --upgrade --quiet
# MAGIC %pip install transformers accelerate safetensors  --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from diffusers import DiffusionPipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
)
base.to(device)

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to(device)

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

# COMMAND ----------

prompt = "A photo of a dog eating dog food"

# run both experts
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]

# COMMAND ----------

image

# COMMAND ----------

import gc

# delete the base and refiner, and free up some memory
del base, refiner
gc.collect()
torch.cuda.empty_cache()
dbutils.library.restartPython()

# COMMAND ----------


