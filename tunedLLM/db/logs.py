import os
import logging
import pandas as pd


class Logs:
    def __init__(self, root_dir: str = None):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.path = f"{root_dir}/logs.csv"
        if os.path.exists(self.path):
            self.log_file = pd.read_csv(self.path)
        else:
            self.log_file = pd.DataFrame(columns=[
                'run_id',
                'user_query',
                'topic',
                'path_to_search_queries',
                'path_to_relevant_papers',
                'path_to_chunks',
                'path_to_qa_pairs',
                'yaml_file_path'
            ])
        self.index = self.log_file.index.max() + 1 if not self.log_file.empty else 0
        logging.info('Initialized logs file client ...')
    
    def update(self, field: str, state):
        self.log_file.loc[self.index, field] = state[field]
    
    def save(self):
        self.log_file.to_csv(self.path, index=False)
        logging.info('Successfully updated logs file ...')
