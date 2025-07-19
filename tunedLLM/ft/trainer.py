import os
import torch
import logging
from typing import Optional
from datetime import datetime
from datasets import load_dataset
from dataclasses import dataclass
from distutils.util import strtobool
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, ModelConfig, SFTConfig, get_peft_config
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None

class Tuner:
    def __init__(
            self,
            model_args: ModelConfig,
            script_args: ScriptArguments,
            training_args: SFTConfig
        ):
        self.model_args = model_args
        self.script_args = script_args
        self.training_args = training_args
    
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

        self.setup()
    
    def get_checkpoint(self, training_args: SFTConfig):
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        return last_checkpoint
    
    def cleanup(sample):
        messages = sample.get("messages")
        if not messages:
            return False
        question = messages[1]
        answer = messages[2]

        if not question or not question['content'].strip():
            return False
        if not answer or not answer['content'].strip():
            return False
        return True
    
    def setup(self):
        self.logger.info(f'Model parameters {self.model_args}')
        self.logger.info(f'Script parameters {self.script_args}')
        self.logger.info(f'Training/evaluation parameters {self.training_args}')

        self.train_dataset = load_dataset('json', data_files=self.script_args.dataset_id_or_path, split='train')
        self.train_dataset = self.train_dataset.filter(self.cleanup)
        self.logger.info(f'Loaded dataset with {len(self.train_dataset)} samples and the following features: {self.train_dataset.features}')


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.script_args.tokenizer_name_or_path if self.script_args.tokenizer_name_or_path else self.model_args.model_name_or_path,
            revision=self.model_args.model_revision,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model_kwargs = dict(
            revision=self.model_args.model_revision,
            trust_remote_code=self.model_args.trust_remote_code,
            torch_dtype=self.model_args.torch_dtype if self.model_args.torch_dtype in ['auto', None] else getattr(torch, self.model_args.torch_dtype),
            use_cache=False if self.training_args.gradient_checkpointing else True,
            low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None
        )

        if self.model_args.load_in_4bit: 
            self.model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=self.model_kwargs['torch_dtype'],
                bnb_4bit_quant_storage=self.model_kwargs['torch_dtype'],
            )
        
        if self.model_args.use_peft:
            self.peft_config = get_peft_config(self.model_args)
        else:
            self.peft_config = None

        self.model = AutoModelForCausalLM.from_pretrained(self.model_args.model_name_or_path, **self.model_kwargs)
        self.training_args.distributed_state.wait_for_everyone()
    
    def tune(self):
        trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            peft_config=self.peft_config,
        )
        if trainer.accelerator.is_main_process and self.peft_config:
            trainer.model.print_trainable_parameters()

        last_checkpoint = self.get_checkpoint(self.training_args)
        if last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
            self.logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')

        self.logger.info(f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {self.training_args.num_train_epochs} epochs***')
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

        metrics = train_result.metrics
        metrics['train_samples'] = len(self.train_dataset)
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
        
        self.logger.info('*** Save model ***')
        if trainer.is_fsdp_enabled and self.peft_config:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type('FULL_STATE_DICT')
        trainer.model.config.use_cache = True
        trainer.save_model(self.training_args.output_dir)
        self.logger.info(f'Model saved to {self.training_args.output_dir}')
        self.training_args.distributed_state.wait_for_everyone()

        self.tokenizer.save_pretrained(self.training_args.output_dir)
        self.logger.info(f'Tokenizer saved to {self.training_args.output_dir}')


        if trainer.accelerator.is_main_process:
            trainer.create_model_card({'tags': ['sft', 'tutorial', 'philschmid']})

        if self.training_args.push_to_hub is True:
            self.logger.info('Pushing to hub...')
            trainer.push_to_hub()

        self.logger.info('*** Training complete! ***')