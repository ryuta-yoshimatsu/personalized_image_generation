#Building Personalized Image Generation Model
Recent advancements in large text-to-image models have shown unparalleled capabilities, enabling high-quality and diverse generation of images based on natural language prompts or existing images. Today, design professionals across various industries are harnessing these models to generate images that serve as inspiration for their next product designs. With further refinement, the generated images may even be used as initial prototypes.

Customization is often necessary for these models. Like any other generative models, tailoring the content is crucial for building a successful application. However, pre-trained text-to-image models frequently lack the capacity to generate specific subjects accurately across various contexts. Thus, fine-tuning the models on images of specific subjects while preserving syntactic and semantic knowledge becomes essential.

The objective of this project is to provide Databricks users with a tool to expedite the development of personalized image generation models. We demonstrate how to fine-tune a text-to-image diffusion model for generation of personalized images in a scalable way. Specifically, we use DreamBooth to fine-tune Stable Diffusion XL using a set of sample images featuring designer chairs.

##Why Stable Diffusion XL?
[Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/) is one of the most powerful open-source text-to-image models available for commercial usage today (as of 2024-03-01). Developed by Stability AI, its weights are publicly accessible via Hugging Face, which has native support on Databricks. Both the off-the-shelf and the fine-tuned versions of the model are widely adopted by numerous companies and are utilized for their mission-critical applications.

##Why DreamBooth?
[Dreambooth](https://dreambooth.github.io/) is an innovative technique developed by researchers from Google Research and Boston University. It enables fine-tuning of text-to-image models using only a few images depicting a particular subject or style. Following fine-tuning, the models gain the ability to generate images of the subject or style in diverse contexts. DreamBooth is seamlessly integrated into Hugging Face’s Diffusers library, with its training scripts readily accessible to the public.


##Why Databricks Mosaic AI?
[Databricks Mosaic AI](https://www.databricks.com/product/machine-learning) offers a great option for GenAI project development and management. It provides a scalable unified platform scoping Data and AI. Data needed to train models is readily available via [Unity Catalog Volumes](https://www.databricks.com/product/unity-catalog) and GenAI models can be easily managed and deployed using [MLflow](https://www.databricks.com/product/managed-mlflow).


##Getting Started
This project is structured in 4 notebooks.  

The first notebook, 00_introduction, walks you through how to download Stable Diffusion XL from Hugging Face and generate an image conditioned on a simple prompt. This notebook is aimed to demonstrate how easy it is to use an open source image generation model off-the-shelf on Databricks. 

The second notebook,  01_preprocessing, downloads a sample training dataset consisting of 25 images of designer chairs from a public repository and applies preprocessing. The main step performed here is to annotate each image with a unique token referring to the subject and a context. We use Unity Catalog Volumes to manage the preprocessed and post-processed images. 

The third notebook, 02_finetuning, shows how to fine-tune Stable Diffusion XL using DreamBooth. Here, we combine with the techniques like mixed precision and LoRA to make the training efficient and to reduce the memory footprint. The second part of the notebook takes the fine-tuned model and registers it to Unity Catalog using MLflow. 
 
The final notebook, 03_deploymet, takes the model registered in Unity Catalog and deploys it behind Databricks Mosaic AI Model Serving endpoint. This allows end users to send an image generation request and get the results back in real time via Rest API.  

To get started, simply clone this repository to your Databricks Repos and run the notebooks in the right sequence. For the compute, we recommend a single node cluster with multiple A10 or A100 GPU instances. In order to use your own images for fine tuning, follow the instructions in the notebook, 01_preprocessing. 

Note that DreamBooth is sensitive to hyperparameters, and it is known to easily overfit. For detailed description of the limitations and how to deal with them, read the original  paper and this blog post. 
