from loguru import logger

from utils_comm.file_util import get_local_ip

model_paths = {
    "Llama-3": "/mnt/nas1/models/meta-llama/Meta-Llama-3-8B-Instruct",
    "openchat-3.5": "/mnt/nas1/models/openchat-3.5-0106",
    "Mistral-7B-Instruct-v0.2": "/mnt/nas1/models/mistralai/Mistral-7B-Instruct-v0.2",
}


def get_model_path(model_name):
    model_path = model_paths[model_name]
    ip = get_local_ip()
    if ip == "123" and model_name == "Llama-3":
        model_path = "/mnt/sde/models/Meta-Llama-3-8B-Instruct"
    logger.info(f"{model_path = }")
    return model_path