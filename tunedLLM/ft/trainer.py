import time
import torch
import shutil
from trl import SFTConfig
from trl import SFTTrainer
import torch.distributed as dist
from datasets import load_dataset
from peft import LoraConfig, PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

class Tuner:
    def __init__(self, root_dir: str = None, local_rank: int = 0):
        self.root = root_dir
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        self.model_id = "google/gemma-3-1b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
    
        if self.model_id == "google/gemma-3-1b-pt":
            self.model_class = AutoModelForCausalLM
        else:
            self.model_class = AutoModelForImageTextToText
        
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
        
        model_kwargs = dict(
            attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
            torch_dtype=torch_dtype # What torch dtype to use, defaults to auto
        )
        
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
            bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
        )
        
        model = self.model_class.from_pretrained(self.model_id, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            modules_to_save=["lm_head", "embed_tokens"]
        )
        
        self.output_dir = f"{root_dir}/tuning/gemma-qlora-energyai"
        args = SFTConfig(
            output_dir=self.output_dir,
            max_seq_length=256,
            packing=True,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            fp16=True if torch_dtype == torch.float16 else False,
            bf16=True if torch_dtype == torch.bfloat16 else False,
            max_grad_norm=0.3,                      
            warmup_ratio=0.03,                    
            lr_scheduler_type="constant",         
            push_to_hub=False,     
            report_to="tensorboard",          
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": True
            }
        )
        
        
        train_dataset = load_dataset('json', data_files=f"{root_dir}/data/dataset.json", split='train')
        train_dataset = train_dataset.filter(self.cleanup)
        
        self.trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            peft_config=peft_config,
            processing_class=tokenizer
        )
    
    def cleanup(self, sample):
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
    
    def tune(self):
        self.trainer.train()
        self.trainer.save_model()
        del self.trainer
        torch.cuda.empty_cache()
        time.sleep(30)
        model = self.model_class.from_pretrained(self.model_id, low_cpu_mem_usage=True)

        peft_config = PeftConfig.from_pretrained(self.output_dir)
        peft_model = PeftModel.from_pretrained(model=model, model_id=self.output_dir, config=peft_config)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(f"{self.root}/tuning/gemma-qlora-energyai-standalone", safe_serialization=True, max_shard_size="2GB")

        processor = AutoTokenizer.from_pretrained(self.output_dir)
        processor.save_pretrained(f"{self.root}/tuning/gemma-qlora-energyai-standalone-tokenizer")
        shutil.rmtree(self.output_dir)