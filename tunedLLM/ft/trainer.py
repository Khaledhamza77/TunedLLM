import os
import logging
from typing import Optional
from datasets import load_dataset
from dataclasses import dataclass
from distutils.util import strtobool
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, TrlParser, ModelConfig, SFTConfig, get_peft_config
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig


@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    spectrum_config_path: Optional[str] = None

class tuner:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
    
    def get_checkpoint(self, training_args: SFTConfig):
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir):
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
        return last_checkpoint
    
    def tune(
            self,
            model_args: ModelConfig,
            script_args: ScriptArguments,
            training_args: SFTConfig
            ):
        self.logger.info(f'Model parameters {model_args}')
        self.logger.info(f'Script parameters {script_args}')
        self.logger.info(f'Training/evaluation parameters {training_args}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            script_args.tokenizer_name_or_path if script_args.tokenizer_name_or_path else model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        self.model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=model_args.torch_dtype if model_args.torch_dtype in ['auto', None] else getattr(torch, model_args.torch_dtype),
            use_cache=False if training_args.gradient_checkpointing else True, # Whether
            low_cpu_mem_usage=True if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) else None
        )