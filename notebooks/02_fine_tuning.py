# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at https://github.com/databricks-industry-solutions.

# COMMAND ----------

# DBTITLE 1,Install requirements and load helper functions
# MAGIC %run ./util

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-tune Stable Diffusion XL with DreamBooth and LoRA

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Set up TensorBoard
# MAGIC
# MAGIC TensorBoard allows you to visualize model training and lets you monitor if the training is going well.
# MAGIC
# MAGIC TensorBoard reads the event log file and exposes it in near real time on the dashboard. But if you're writing out the event log to DBFS, it won't show until the file is closed for writing, which is when the training is complete. This is not good for a real time monitoring purpose. In this case, we suggest to write the event log to the driver node (instead of DBFS) and run your TensorBoard there. Files stored on the driver node may get removed when the cluster terminates or restarts. But when you are running the training on Databricks notebook, MLflow will automatically log your Tensorboard artifacts, and you will be able to recover them later. You can find the example of this below.

# COMMAND ----------

import os
from tensorboard import notebook

logdir = "/databricks/driver/logdir/sdxl/"
notebook.start("--logdir {} --reload_multifile True".format(logdir))

# COMMAND ----------

theme = "chair"
catalog = "sdxl_image_gen"
volumes_dir = "/Volumes/sdxl_image_gen"
os.environ["DATASET_NAME"] = f"{volumes_dir}/{theme}/dataset"
os.environ["OUTPUT_DIR"] = f"{volumes_dir}/{theme}/adaptor"
os.environ["LOGDIR"] = logdir
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{theme}.adaptor")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set Parameters
# MAGIC To ensure we can use DreamBooth with LoRA on a heavy pipeline like Stable Diffusion XL, we're using the following hyperparameters:
# MAGIC
# MAGIC * Gradient checkpointing (`--gradient_accumulation_steps`)
# MAGIC * 8-bit Adam (`--use_8bit_adam`)
# MAGIC * Mixed-precision training (`--mixed-precision="fp16"`)
# MAGIC * Some other parameters are defined in `yamls/accelerate/zero2.yaml`
# MAGIC <br>
# MAGIC
# MAGIC Other parameters:
# MAGIC * Use `--output_dir` to specify your LoRA model repository name.
# MAGIC * Use `--caption_column` to specify name of the caption column in your dataset.
# MAGIC * Make sure to pass the right number of GPUs to the parameter `num_processes` in `yamls/accelerate/zero2.yaml`.

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file ../yamls/accelerate/zero2.yaml ../personalized_image_generation/train_dreambooth_lora_sdxl.py \
# MAGIC   --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
# MAGIC   --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
# MAGIC   --dataset_name=$DATASET_NAME \
# MAGIC   --caption_column="caption" \
# MAGIC   --instance_prompt="" \
# MAGIC   --output_dir=$OUTPUT_DIR \
# MAGIC   --resolution=1024 \
# MAGIC   --train_batch_size=1 \
# MAGIC   --gradient_accumulation_steps=3 \
# MAGIC   --gradient_checkpointing \
# MAGIC   --learning_rate=1e-4 \
# MAGIC   --snr_gamma=5.0 \
# MAGIC   --lr_scheduler="constant" \
# MAGIC   --lr_warmup_steps=0 \
# MAGIC   --use_8bit_adam \
# MAGIC   --max_train_steps=500 \
# MAGIC   --checkpointing_steps=717 \
# MAGIC   --seed="0" \
# MAGIC   --report_to="tensorboard" \
# MAGIC   --logging_dir=$LOGDIR

# COMMAND ----------

# MAGIC %sh ls -ltr $OUTPUT_DIR

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test inference

# COMMAND ----------

from diffusers import DiffusionPipeline, AutoencoderKL
import torch

theme = "chair"
catalog = "sdxl_image_gen"
volumes_dir = "/Volumes/sdxl_image_gen"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.load_lora_weights(f"{volumes_dir}/{theme}/adaptor/pytorch_lora_weights.safetensors")
pipe = pipe.to(device)

# COMMAND ----------

import os
import glob

types = os.listdir("../images/chair")
num_imgs_to_preview = len(types)
imgs = []
for type in types:
    imgs.append(
        pipe(
            prompt=f"A photo of a red {type} chair in a living room",
            num_inference_steps=25,
        ).images[0]
    )
show_image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model to MLflow

# COMMAND ----------

import mlflow
import torch

class sdxl_fine_tuned(mlflow.pyfunc.PythonModel):
    def __init__(self, vae_name, model_name):
        self.vae_name = vae_name
        self.model_name = model_name

    def load_context(self, context):
        """
        This method initializes the vae and the model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = diffusers.AutoencoderKL.from_pretrained(
            self.vae_name, torch_dtype=torch.float16
        )
        self.pipe = diffusers.DiffusionPipeline.from_pretrained(
            self.model_name,
            vae=self.vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.load_lora_weights(context.artifacts["repository"])
        self.pipe = self.pipe.to(self.device)

    def predict(self, context, model_input):
        """
        This method generates output for the given input.
        """
        prompt = model_input["prompt"][0]
        num_inference_steps = model_input.get("num_inference_steps", [25])[0]
        # Generate the image
        image = self.pipe(
            prompt=prompt, num_inference_steps=num_inference_steps
        ).images[0]
        # Convert the image to numpy array for returning as prediction
        image_np = np.array(image)
        return image_np


# COMMAND ----------

theme = "chair"
catalog = "sdxl_image_gen"
volumes_dir = "/Volumes/sdxl_image_gen"
vae_name = "madebyollin/sdxl-vae-fp16-fix"
model_name = "stabilityai/stable-diffusion-xl-base-1.0"
output = f"{volumes_dir}/{theme}/adaptor/pytorch_lora_weights.safetensors"

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec, TensorSpec
import transformers, bitsandbytes, accelerate, deepspeed, diffusers

mlflow.set_registry_uri("databricks-uc")

# Define input and output schema
input_schema = Schema(
    [ColSpec(DataType.string, "prompt"), ColSpec(DataType.long, "num_inference_steps")]
)
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, 768, 3))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame(
    {"prompt": [f"A photo of a {theme} in a living room"], "num_inference_steps": [25]}
)

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=sdxl_fine_tuned(vae_name, model_name),
        artifacts={"repository": output},
        pip_requirements=[
            "transformers==" + transformers.__version__,
            "bitsandbytes==" + bitsandbytes.__version__,
            "accelerate==" + accelerate.__version__,
            "deepspeed==" + deepspeed.__version__,
            "diffusers==" + diffusers.__version__,
        ],
        input_example=input_example,
        signature=signature,
    )
    mlflow.set_tag("dataset", f"{volumes_dir}/{theme}/dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog

# COMMAND ----------

# Make sure that the schema for the model exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.model")
# Register the model 
registered_name = f"{catalog}.model.sdxl-fine-tuned-{theme}"
result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model",
    registered_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the registered model back to make inference
# MAGIC
# MAGIC Restart the Python to release the GPU memory occupied in Training.

# COMMAND ----------

def get_latest_model_version(mlflow_client, registered_name):
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

import mlflow
from mlflow import MlflowClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

theme = "chair"
catalog = "sdxl_image_gen"
volumes_dir = "/Volumes/sdxl_image_gen"
registered_name = f"{catalog}.model.sdxl-fine-tuned-{theme}"
model_version = get_latest_model_version(mlflow_client, registered_name)
logged_model = f"models:/{registered_name}/{model_version}"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# Use any of the following token to generate personalized images: 'bcnchr', 'emslng', 'hsmnchr', 'rckchr', 'wdnchr'
input_example = pd.DataFrame(
    {
        "prompt": ["A photo of a brown emslng chair in a living room"],
        "num_inference_steps": [25],
    }
)
image = loaded_model.predict(input_example)
show_image(image)

# COMMAND ----------

# Assign an alias to the model
mlflow_client.set_registered_model_alias(registered_name, "champion", model_version)

# COMMAND ----------


