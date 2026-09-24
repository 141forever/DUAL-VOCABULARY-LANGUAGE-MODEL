import torch
import time
import pdb
import os
import re
import random
from collections import Counter
from ast import literal_eval
import torch.nn as nn
from vllm import LLM, SamplingParams
from functools import partial

from datasets import load_dataset,Dataset
from trl.trl.experimental.gold import GOLDConfig, GOLDTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, EvalPrediction, PreTrainedTokenizer
from peft import LoraConfig


device = "cuda" if torch.cuda.is_available() else "cpu"



from dataclasses import dataclass
from typing import Any, cast
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForGOLD:
    """
    Data collator for ChatML format datasets for the GOLD trainer.
    Expect `messages` for each example.
    """

    student_tokenizer: PreTrainedTokenizerBase
    teacher_tokenizer: PreTrainedTokenizerBase
    student_chat_template_kwargs: dict | None = None
    teacher_chat_template_kwargs: dict | None = None
    ignore_index: int = -100

    def __post_init__(self):
        assert self.student_tokenizer.pad_token_id is not None
        assert self.teacher_tokenizer.pad_token_id is not None

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_teacher_input_ids = []
        batch_attention_masks = []
        batch_teacher_attention_masks = []
        batch_targets = []

        for example in examples:
            prompt_messages = example["messages"]
            # Remove completion
            while prompt_messages[-1]["role"] == "assistant":
                prompt_messages.pop()

            input_ids = self.student_tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **(self.student_chat_template_kwargs or {}),
            )
            input_ids = input_ids['input_ids'][0]
            teacher_input_ids = self.teacher_tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **(self.teacher_chat_template_kwargs or {}),
            )
            teacher_input_ids = teacher_input_ids['input_ids'][0]
            batch_input_ids.append(input_ids)
            batch_teacher_input_ids.append(teacher_input_ids)
            batch_attention_masks.append(torch.ones_like(input_ids))
            batch_teacher_attention_masks.append(torch.ones_like(teacher_input_ids))

            # Labels (concat nums + target)
            if "target" in example and "nums" in example:
                targets = torch.LongTensor([example["target"], *example["nums"]])
                batch_targets.append(targets)

        input_ids = pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.student_tokenizer.pad_token_id,  # type: ignore
            padding_side="left",
        )
        teacher_input_ids = pad_sequence(
            batch_teacher_input_ids,
            batch_first=True,
            padding_value=self.teacher_tokenizer.pad_token_id,  # type: ignore
            padding_side="left",
        )
        attention_masks = pad_sequence(
            batch_attention_masks,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        teacher_attention_masks = pad_sequence(
            batch_teacher_attention_masks,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )

        batch = {
            "input_ids": input_ids,
            "teacher_input_ids": teacher_input_ids,
            "attention_masks": attention_masks,
            "teacher_attention_masks": teacher_attention_masks,
        }

        if batch_targets:
            batch["targets"] = pad_sequence(batch_targets,batch_first=True,padding_value=-100)

        return batch

def prepare_training_data_and_model():
    
    student_name = "Llama-3.2-1B-Instruct"
    teacher_name = "Qwen3-4B"
    new_head_path = "../proj_final.pt" 

    tokenizer = AutoTokenizer.from_pretrained(student_name,local_files_only=True)
    if tokenizer.pad_token is None: 
        if ('Llama' in student_name or 'llama' in student_name) and ('qwen' not in student_name or 'Qwen' not in student_name):
            tokenizer.pad_token = "<|end_of_text|>"
            tokenizer.pad_token_id = 128001
        else:
            tokenizer.pad_token = tokenizer.eos_token
        
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name,local_files_only=True)
    assert teacher_tokenizer.pad_token is not None
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(student_name,local_files_only=True).to(device)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_name,local_files_only=True).to(device)

    weight = torch.load(new_head_path, map_location="cpu")["weight"]
    new_lm_head = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        new_lm_head.weight.copy_(weight)
    teacher_model.lm_head = new_lm_head.to(
        device=device,
        dtype=teacher_model.dtype,
    )
    
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    train_dataset = Dataset.from_parquet(
        "xxx.parquet" 
    )
    train_dataset = train_dataset.map(lambda x: {"dataset_name": "our data"})
    
    test_dataset = Dataset.from_parquet("xxx.parquet")
    test_dataset_converted = []
    for d in test_dataset:
        dd = {}
        dd['messages'] = d['prompt']
        dd['target'] = d['target']
        dd['nums'] = d['nums']
        test_dataset_converted.append(dd)
    test_dataset_converted = random.sample(test_dataset_converted, 1000)
    test_dataset_converted = Dataset.from_list(test_dataset_converted)
    
    return student_name, teacher_name, model, teacher_model, train_dataset, test_dataset_converted, tokenizer, teacher_tokenizer

def set_training_settings_and_train(student_name, teacher_name, model, teacher_model, train_dataset, test_dataset_converted, tokenizer, teacher_tokenizer):
    
    changed_teacher_tokenizer = "Qwen3-4B"
    
    training_args = GOLDConfig(
        ### train
        learning_rate=2e-6, 
        warmup_ratio=0.01,
        per_device_train_batch_size=21, 
        max_completion_length = 1024,
        max_length = 1024+1024, 
        teacher_model_name_or_path=teacher_name,
        teacher_tokenizer_name_or_path=teacher_name,
        student_model_name_or_path=student_name,
        student_tokenizer_name_or_path=student_name,
        bf16=True,
        use_uld_loss=True,
        use_extended_uld=True,
        uld_use_hybrid_loss=True,
        lr_scheduler_type = 'cosine', #'cosine_with_min_lr' / 'cosine'
        num_train_epochs=2, 
        gradient_accumulation_steps=1,
        lmbda = 1.0,
        beta = 0.0,
        uld_crossentropy_weight = 0.0,
        uld_distillation_weight = 1.0,
        ### vllm
        use_vllm = True,
        vllm_mode = "server",
        vllm_server_host = '127.0.0.1',
        vllm_server_port = 8000, 
        vllm_gpu_memory_utilization = 0.8,
        ### eval
        do_eval=False,
        eval_strategy="steps",
        eval_steps=100,
        per_device_eval_batch_size=128,
        label_names=["targets"],
        eval_on_start=False,
        ### log
        logging_steps=10,
        save_strategy="steps",
        save_steps=10, 
        report_to=[],
        push_to_hub=False,
    )
    
    training_args.past_index = -1

    trainer = GOLDTrainer(
        model=model,
        teacher_model=teacher_model,
        changed_teacher_tokenizer=changed_teacher_tokenizer,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset_converted,
        data_collator=DataCollatorForGOLD(
        student_tokenizer=tokenizer,
        teacher_tokenizer=teacher_tokenizer),
    )
    trainer.create_model_card = lambda *args, **kwargs: None 
    trainer.train()
    return trainer, tokenizer




if __name__ == "__main__":
    student_name, teacher_name, model, teacher_model, train_dataset, test_dataset_converted, tokenizer, teacher_tokenizer = prepare_training_data_and_model()
    trainer, tokenizer = set_training_settings_and_train(student_name, teacher_name, model, teacher_model, train_dataset, test_dataset_converted, tokenizer, teacher_tokenizer)
    
