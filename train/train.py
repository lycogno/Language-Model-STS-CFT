import os
import torch
import transformers
from typing import Optional
from datasets import load_from_disk
from dataclasses import dataclass, field
from contrastive_trainer import ContrastiveTrainer
from transformers import AutoModelForCausalLM, HfArgumentParser, set_seed

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataArguments:
    train_data_path: str = field(
        metadata={"help": "Path to training data"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    temperature: Optional[float] = field(default=0.05)

def main(model_args, data_args, training_args):
    set_seed(training_args.seed)

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                                torch_dtype=torch.bfloat16,
                                                trust_remote_code=True,
                                                local_files_only=True)

    # Data
    train_dataset = load_from_disk(data_args.train_data_path)

    trainer = ContrastiveTrainer(model=model,
                                args=training_args,
                                train_dataset=train_dataset)

    trainer.accelerator.print(f"{trainer.model}")

    # Train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # Saving final model
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "minicpm-dense-retrieval"
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)