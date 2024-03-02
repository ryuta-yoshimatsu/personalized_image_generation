# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at https://github.com/databricks-industry-solutions.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Building Personalized Image Generation Model
# MAGIC Recent advancements in large text-to-image models have shown unparalleled capabilities, enabling high-quality generation of images in diverse contexts based on natural language prompts or existing images. Today, design professionals across various industries are harnessing these models to generate images that serve as inspiration for their next product designs. With or without further refinement, the generated images may even be used as initial prototypes.
# MAGIC
# MAGIC This solution accelerator provides Databricks users with a tool to expedite the end-to-end development of personalized image generation applications. The asset including a series of notebooks demonstrates how to preprocess training images, fine-tune a text-to-image diffusion model, manage the fine-tuned model, and deploy the model behind an endpoint and make it available for downstream applications. The solution is by design customizable (bring your own images) and scalable leveraging Databricks powerful distributed compute.

# COMMAND ----------

# DBTITLE 1,Install requirements and load helper functions
# MAGIC %run ./util

# COMMAND ----------

# MAGIC %md
# MAGIC To use the off-the-shelf model, you can run:

# COMMAND ----------

import torch
from diffusers import DiffusionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.to(device)

# COMMAND ----------

prompt = "A photo of a brown leather chair in a living room."
image = pipe(prompt=prompt).images[0]
show_image(image)

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | hugging face | | |
# MAGIC | stable diffusion xl base | | |
# MAGIC | dreambooth | | |
