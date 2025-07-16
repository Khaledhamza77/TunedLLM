import os
import logging
import pandas as pd
from .gemma.agent import Gemma

class master:
    def __init__(self, root_dir: str):
        self.root = root_dir
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        os.makedirs(f"{self.root}/data/jobs", exist_ok=True)
    
    def batch_data(self, data_path):
        df = pd.read_parquet(data_path)