# Databricks notebook source
# Run IN the FEVM workspace (serverless). Serverless notebooks don't bundle
# mlflow; we also upgrade typing_extensions and add huggingface_hub (used to
# bundle GDINO/CLIP weights as artifacts since serving has no internet egress).
# MAGIC %pip install -q -U typing_extensions mlflow databricks-sdk huggingface_hub pandas pillow
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
import os, sys
os.environ["UC_CATALOG"] = "serverless_stable_lsryzn_catalog"
os.environ["UC_SCHEMA"] = "image_masking"
os.environ["ENDPOINT_NAME"] = "image-masking-vision"
os.environ["WORKLOAD_TYPE"] = "GPU_SMALL"
ROOT = "/Workspace/Users/brandon.cowen@databricks.com/image_masking_v5deploy"
sys.path.insert(0, ROOT)
from deploy.serving.deploy_vision_endpoint import main
main()
