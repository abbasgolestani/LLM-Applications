# ML Accelerator LLM Edition
# Lab 8: PEFT (Parameter Efficient Fine Tuning)
# PEFT fine tuning of Llama2 7B model with public dataset for summarization

# Step 1 import libraries
import os
from functools import lru_cache

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler

# Step 2 Download model and samsum and alpaca public dataset
@lru_cache
def get_path_manager():
    path_manager = PathManager()
    path_manager.register_handler(ManifoldPathHandler())
    return path_manager


def download_manifold_files(
    bucket_name,
    file_paths,
    local_path,
):
    path_manager = get_path_manager()
    base_manifold_path = f"manifold://{bucket_name}/tree/"

    for file_path in file_paths:
        dir_path = os.path.join(local_path, os.path.dirname(file_path))
        # Create path if not exist
        if not os.path.exists(dir_path):
            print(f"Make dirs '{dir_path}'")
            os.makedirs(dir_path)
        else:
            print(f"'{dir_path}' exists")

        source = os.path.join(base_manifold_path, file_path)
        dest = os.path.join(local_path, file_path)
        if not os.path.exists(dest):
            print(f"Copying source '{source}' to dest '{dest}'")
            path_manager.copy(source, dest)


def download_manifold_dir(
    bucket_name,
    dir_paths,
    local_path,
):
    path_manager = get_path_manager()
    base_manifold_path = f"manifold://{bucket_name}/tree/"
    for dir_path in dir_paths:
        # Create path if not exist
        local_dir_path = os.path.join(local_path, dir_path)
        if not os.path.exists(local_dir_path):
            print(f"Make dirs '{local_dir_path}'")
            os.makedirs(local_dir_path)
        else:
            print(f"'{local_dir_path}' exists")

        current_dir = os.path.join(base_manifold_path, dir_path)
        print(f"Download dir '{current_dir}'")
        for filename in path_manager.ls(current_dir):
            source = os.path.join(current_dir, filename)
            dest = os.path.join(local_dir_path, filename)
            if not os.path.exists(dest):
                print(f"Copying source '{source}' to dest '{dest}'")
                path_manager.copy(source, dest)

model_local_dir = "/tmp/"
model_name = "models/pretrained/Llama-2-7b-chat-hf"
model_local_path = os.path.join(model_local_dir, model_name)

download_manifold_dir(
    bucket_name = "biz_rnd_llms",
    dir_paths = [model_name],
    local_path = model_local_dir,
)

finetune_dataset = "datasets/summarize/samsum/corpus"
download_manifold_dir(
    bucket_name = "biz_rnd_llms",
    dir_paths = [finetune_dataset],
    local_path = model_local_dir,
)
samsum_finetune_dataset_local_path = os.path.join(model_local_dir, finetune_dataset)

input_data_local_dir = "/tmp/"
alpaca_remote_path = "datasets/finetuning/alpaca_data_cleaned.json"
alpaca_finetune_dataset_local_path = os.path.join(
    input_data_local_dir, alpaca_remote_path
)
download_manifold_files(
    bucket_name = "biz_rnd_llms",
    file_paths = [alpaca_remote_path],
    local_path = model_local_dir,
)


# Step 3 Load pre-trained model
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
model_id=model_local_path
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)


# Step 4 Pre process samsum dataset for fine tuning.
from dataclasses import dataclass
from functools import partial

from llama_recipes.datasets.alpaca_dataset import InstructionDataset

def get_preprocessed_dataset(
    tokenizer, dataset_config,  split: str = "train"
) -> torch.utils.data.Dataset:
    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    return dataset_config.loader(
        dataset_config,
        dataset_config,
        tokenizer,
    )

@dataclass
class alpaca_dataset_config:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = alpaca_finetune_dataset_local_path
    file_type: str = "json"
    loader = partial(InstructionDataset)
alpaca_train_dataset: torch.utils.data.Dataset = get_preprocessed_dataset(tokenizer, alpaca_dataset_config, 'train')

#------------
import datasets
def get_preprocessed_samsum(dataset, dataset_config, tokenizer, split):
    max_length = 512
    prompt = (
        f"Summarize this dialog:\n{{dialog}}\n---\nSummary:\n" # noqa
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["dialogue"]),
            "summary": sample["summary"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False, truncation=True, padding='max_length', max_length=max_length, return_attention_mask=True)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False, truncation=True, padding='max_length', max_length=max_length, return_attention_mask=True)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
@dataclass
class samsum_dataset_config:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = samsum_finetune_dataset_local_path
    file_type: str = "json"
    loader = get_preprocessed_samsum
samsum_train_dataset = datasets.load_dataset("json", data_files=f"{samsum_finetune_dataset_local_path}/train.json")['train']
samsum_train_dataset = get_preprocessed_samsum(samsum_train_dataset, samsum_dataset_config, tokenizer, 'train')


# Step 5 Evaluate base model
eval_prompt = """
Summarize this dialog:
A: Hi Tom, are you busy tomorrow’s afternoon?
B: I’m pretty sure I am. What’s up?
A: Can you go with me to the animal shelter?.
B: What do you want to do?
A: I want to get a puppy for my son.
B: That will make him so happy.
A: Yeah, we’ve discussed it many times. I think he’s ready now.
B: That’s good. Raising a dog is a tough issue. Like having a baby ;-)
A: I'll get him one of those little dogs.
B: One that won't grow up too big;-)
A: And eat too much;-))
B: Do you know which one he would like?
A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
B: I bet you had to drag him away.
A: He wanted to take it home right away ;-).
B: I wonder what he'll name it.
A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
---
Summary:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))


# Step 6 Create PEFT Configuration for fine tuning model

model.train()

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        prepare_model_for_int8_training,
        TaskType,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# create peft config
model, lora_config = create_peft_config(model)


#-----------
from contextlib import nullcontext

from transformers import TrainerCallback
enable_profiler = False
output_dir = "/tmp/llama-output"

config = {
    'lora_config': lora_config,
    'learning_rate': 1e-4,
    'num_train_epochs': 1,
    'gradient_accumulation_steps': 2,
    'per_device_train_batch_size': 2,
    'gradient_checkpointing': False,
}

# Set up profiler
if enable_profiler:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)

    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler

        def on_step_end(self, *args, **kwargs):
            self.profiler.step()

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()


#Step 8 Fine Tune model
# Note: This step takes 15 hours on a server with P100
rom transformers import default_data_collator, Trainer, TrainingArguments



# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    #bf16=False,  # Use BF16 if available. Doesn't work on a V100.
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch_fused",
    max_steps=total_steps if enable_profiler else -1,
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=samsum_train_dataset,
        data_collator=default_data_collator,
        callbacks=[profiler_callback] if enable_profiler else [],
    )

    # Start training
    trainer.train()



# Step 9 Save fine tuned model to local directory.
model.save_pretrained(output_dir)
model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))


# Step 10 Upload fine tuned model to your user directory in manifold
def upload_dir_manifold(
    bucket_name,
    project_name,
    dir_paths,
    local_path,
):
    path_manager = get_path_manager()
    base_manifold_path = f"manifold://{bucket_name}/tree/uploads/{os.environ.get('USER')}/{project_name}/"
    for dir_path in dir_paths:
        # Create path if not exist
        local_dir_path = os.path.join(local_path, dir_path)

        remote_dir_path = os.path.join(base_manifold_path, dir_path)
        print(f"Upload '{local_dir_path}' to '{remote_dir_path}'")
        for filename in os.listdir(local_dir_path):
            filename_path = os.path.join(local_dir_path, filename)
            if not os.path.isfile(filename_path):
                continue
            dest = os.path.join(remote_dir_path, filename)
            source = filename_path
            if not path_manager.exists(dest):
                if not path_manager.exists(remote_dir_path):
                    path_manager.mkdirs(remote_dir_path)
                print(f"Copying source '{source}' to dest '{dest}'")
                path_manager.copy(source, dest)
upload_dir_manifold(
    bucket_name = "biz_rnd_llms",
    project_name = "finetune_summary_1",
    dir_paths = ".",
    local_path = output_dir,
)


# Step 11 Evaluate fine tuned model
from peft import PeftModel

tokenizer = LlamaTokenizer.from_pretrained(model_id)

finetuned_model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)

lora_path = "/tmp/llama-output/"
finetuned_model = PeftModel.from_pretrained(finetuned_model, lora_path, device_map='auto', torch_dtype=torch.float16)
finetuned_model = finetuned_model.merge_and_unload()
finetuned_model.eval()
with torch.no_grad():
    print(tokenizer.decode(finetuned_model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))










