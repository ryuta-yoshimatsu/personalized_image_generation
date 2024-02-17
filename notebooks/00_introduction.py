# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at https://github.com/databricks-industry-solutions.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Building Personalized Image Generation Model
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Why Stable Diffusion XL?
# MAGIC [Stable Diffusion XL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)   
# MAGIC
# MAGIC ## Why Dreambooth?
# MAGIC [Dreambooth](https://dreambooth.github.io/)
# MAGIC
# MAGIC ## Why Databricks Mosaic AI?
# MAGIC [Databricks Mosaic AI](https://www.databricks.com/product/machine-learning) offers a great option for GenAI project development and management. It provides a scalable unified platform scoping Data and AI. Data needed to train models is readily available via [Unity Catalog Volumes](https://www.databricks.com/product/unity-catalog) and GenAI models can be easily managed and deployed using [MLflow](https://www.databricks.com/product/managed-mlflow).

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
  variant="fp16"
)

pipe.to(device)

# COMMAND ----------

prompt = "A photo of a chair in a living room."
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

# COMMAND ----------


