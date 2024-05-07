# You need the following libraries 
# transformers == 4.32.0
# bitsandbytes == 0.41.0
# auto-gptq == 0.4.2
# optimum == 1.12.0
# https://huggingface.co/blog/overview-quantization-transformers#:~:text=We%20saw%20that%20bitsandbytes%20is,and%20fine%2Dtune%20the%20adapters

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import seaborn as sns

N_BATCHES = 10
MAX_NEW_TOKENS = 30
BATCH_SIZE = [1, 2, 4, 8, 16, 32]

bnb_model_id = "meta-llama/Llama-2-7b-hf"
gptq_model_id = "TheBloke/Llama-2-7B-GPTQ"

def warmup_and_benchmark(model, inputs):
    _ = model.generate(**inputs, max_new_tokens=20, eos_token_id=-1)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(N_BATCHES):
        _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=-1)
    end_event.record()
    torch.cuda.synchronize()

    return (start_event.elapsed_time(end_event) * 1.0e-3) / N_BATCHES

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(bnb_model_id)

bnb_model = AutoModelForCausalLM.from_pretrained(
    bnb_model_id, 
    quantization_config=quantization_config, 
    device_map={"":0}, 
    use_auth_token=True
)
gptq_model = AutoModelForCausalLM.from_pretrained(
    gptq_model_id, 
    device_map={"":0}
)

bnb_total_time_dict = {}
gptq_total_time_dict = {}

for batch_size in tqdm(BATCH_SIZE):
    text = [
        "hello"
    ] * batch_size
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")

    # warmup
    bnb_timing = warmup_and_benchmark(bnb_model, inputs)
    bnb_total_time_dict[f"{batch_size}"] = bnb_timing

    gptq_timing = warmup_and_benchmark(gptq_model, inputs)
    gptq_total_time_dict[f"{batch_size}"] = gptq_timing


sns.set(style="darkgrid")
# plot both lines
sns.lineplot(data=bnb_total_time_dict, color="blue", label="bitsandbytes-QLoRA")
sns.lineplot(data=gptq_total_time_dict, color="orange", label="GPTQ-4bit")

plt.ylabel("Average inference time (s)")
plt.xlabel("Batch size")
plt.title("Comparing average inference time between bnb-4bit model vs GPTQ model", fontsize = 8)

plt.legend()

# save plot
plt.savefig("seaborn_comparison_plot.jpg", dpi=300)