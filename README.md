---
library_name: peft
---
## Training procedure


The following `bitsandbytes` quantization config was used during training:
- quant_method: bitsandbytes
- load_in_8bit: True
- load_in_4bit: False
- llm_int8_threshold: 6.0
- llm_int8_skip_modules: None
- llm_int8_enable_fp32_cpu_offload: False
- llm_int8_has_fp16_weight: False
- bnb_4bit_quant_type: fp4
- bnb_4bit_use_double_quant: False
- bnb_4bit_compute_dtype: float32
### Framework versions


- PEFT 0.6.0.dev0
```
import torch
from peft import PeftModel
import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
 
model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, "my-llm", torch_dtype=torch.float16)
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
 
model = model.eval()
model = torch.compile(model)
PROMPT_TEMPLATE = f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
 
### Instruction:
[INSTRUCTION]
 
### Response:
"""

def create_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.replace("[INSTRUCTION]", instruction)
 
print(create_prompt("What is (are) Glaucoma ?"))

def generate_response(prompt: str, model: PeftModel) -> GreedySearchDecoderOnlyOutput:
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)
 
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        repetition_penalty=1.1,
    )
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )
def format_response(response: GreedySearchDecoderOnlyOutput) -> str:
    decoded_output = tokenizer.decode(response.sequences[0])
    response = decoded_output.split("### Response:")[1].strip()
    return "\n".join(textwrap.wrap(response))

def ask_alpaca(prompt: str, model: PeftModel = model) -> str:
    prompt = create_prompt(prompt)
    response = generate_response(prompt, model)
    print(format_response(response))
ask_alpaca("What is (are) Glaucoma ?")
```
```
autotrain llm --train --project_name my-llm --model meta-llama/Llama-2-7b-hf --data_path "data" --train_split "train" --text_column "text" --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 10 --num_train_epochs 3 --trainer sft
 --use_flash_attention_2
```
https://www.mlexpert.io/machine-learning/tutorials/alpaca-and-llama-inference
