import os
import glob
import json
import logging
import subprocess
import pandas as pd
from tqdm import tqdm
import concurrent.futures


class LLMSwarm:
    def __init__(
            self,
            model_name: str = None,
            user_query: str = None,
            root_dir: str = None, 
            chunk_scoring: str = None,
            chunk_to_qa: str = None,
            train_prompt: str = None
        ):
        self.root = root_dir
        self.chunk_scoring = chunk_scoring
        self.chunk_to_qa = chunk_to_qa
        self.user_query = user_query
        self.model_name = model_name
        self.n_jobs = 40
        self.train_prompt = train_prompt
        self.ports = list(range(11434, 11474))
        self.executables = []
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def run(self, qa=False):
        self.parallelize(qa=qa)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(self.run_script, script): script for script in self.executables}
            logging.info('Started chunking process ...')
            for future in concurrent.futures.as_completed(futures):
                script = futures[future]
                try:
                    data = future.result()
                    if "error" in data:
                        logging.warning(f"{script} completed with output:\n{data}")
                except Exception as exc:
                    logging.error(f"{script} generated an exception: {exc}")
        self.concat_batches()
    
    def run_script(self, script):
        result = subprocess.Popen(
            ['python', '-u', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in result.stdout:
            if "HTTP Request:" in line:
                continue 
            elif 'ERROR' in line:
                logging.error(line.strip())
            elif 'Progress update:' in line:
                logging.info(line.strip())

        returncode = result.wait()
        if returncode != 0:
            logging.error(f"Script {script} failed with return code {returncode}.")
            return {"error": f"Script {script} failed with return code {returncode}."}
        return {"status": "success"}

    def concat_batches(self):
        if self.chunk_scoring:
            parquet_files = sorted(glob.glob(f"{self.root}/chunks_*.parquet"))
            if not parquet_files:
                logging.warning(f"No chunks parquet files found to concatenate.")
                return

            df_list = [pd.read_parquet(f) for f in parquet_files]
            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df.reset_index(drop=True, inplace=True)
            output_path = f"{self.root}/chunks.parquet"
            try:
                combined_df.to_parquet(output_path, index=False)
                logging.info(f"Combined chunks saved to {output_path}")
            except Exception as e:
                logging.error(f"Failed to save chunks: {e}")
                return

            for f in parquet_files:
                try:
                    os.remove(f)
                except Exception as e:
                    logging.error(f"Failed to delete {f}: {e}")
            return output_path
        elif self.chunk_to_qa:
            json_files = sorted(glob.glob(f"{self.root}/dataset_*.json"))
            if not json_files:
                logging.warning(f"No training batches found to concatenate")
                return
            training_dataset = []
            for jsnf in json_files:
                try:
                    with open(jsnf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        training_dataset.extend(data)
                except Exception as e:
                    logging.error("Failed to concatenate json files: " + str(e))
            path = f"{self.root}/dataset.json"
            try:
                training_dataset.to_json(path, orient="records")
                logging.info(f"combined training dataset saved to {path}")
            except Exception as e:
                logging.error(f"Failed to write json training set: {e}")
                return
            for f in json_files:
                os.remove(f)
            return path

    def parallelize(self, qa=False):
        if self.chunk_scoring:
            data_path = self.chunk_scoring
            logging.info("Batching metadata for chunking and scoring...")
        elif self.chunk_to_qa:
            data_path = self.chunk_to_qa
            logging.info("Batching chunks for q/a pair generation...")
        df = pd.read_parquet(data_path)
        if qa:
            df = df[df['relevance_class'] == 'a']
        num_rows = len(df)
        rows_per_split = num_rows // self.n_jobs
        remainder = num_rows % self.n_jobs
        start_idx = 0
        self.setup_worker()
        for i in tqdm(range(self.n_jobs)):
            os.makedirs(f"{self.root}/jobs/batch_{i}", exist_ok=True)
            end_idx = start_idx + rows_per_split
            if i == self.n_jobs - 1:
                end_idx += remainder
            df2 = df.iloc[start_idx:end_idx]
            df2.reset_index(inplace=True, drop=True)
            df2.to_parquet(
                f'{self.root}/jobs/batch_{i}/data.parquet'
            )
            start_idx = end_idx
            try:
                self.write_worker(i)
                logging.info(f"worker {i} created successfully.")
            except Exception as e:
                logging.error(f"worker {i} was not created: {e}")
            del df2
        del df
        logging.info('Successfully batched metadata and created workers.')
    
    def write_worker(self, i: int):
        worker_path = f"{self.root}/jobs/batch_{i}/worker.py"
        self.executables.append(worker_path)
        with open(worker_path, "w") as worker:
            worker.write(self.worker_script.format(self=self, i=i, port=self.ports[i]))
    
    def setup_worker(self):
        self.worker_script = """#Auto-generated worker script
import logging
import pandas as pd
from tunedLLM.llm.agent import LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("worker " + str({i}) + " started")
    worker_dir = f"{self.root}/jobs/batch_{i}"
    df = pd.read_parquet(worker_dir + "/data.parquet"); total = len(df)
    llm = LLM(root_dir=None, model_name="{self.model_name}", port={port})"""
        if self.chunk_scoring:
            self.worker_script += """
    result = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    for idx, row in df.iterrows():
        with open(f"{self.root}/full_texts/"+str(row['id'])+".txt", 'r', encoding='utf-8') as f:
            full_text = f.read()
        chunks = text_splitter.split_text(full_text)
        if idx % 2 == 0 and idx != 0:
            print('Worker '+str({i})+' Progress update: '+str(idx)+'/'+str(total))
        for chk_idx, chunk in enumerate(chunks):
            result.append(
                dict(
                    chunk_id = str(row['id']) + "_" + str(chk_idx),
                    id = str(row["id"]),
                    chunk = chunk,
                    relevance_class = llm.score_chunk('{self.user_query}', chunk, row),
                    title = row['title'],
                    abstract = row['abstract']
                )
            )
    try:
        result_df = pd.DataFrame(result); del result
        result_df.to_parquet(f"{self.root}/chunks_{i}.parquet", index=False)
        logging.info("Chunking completed.")
    except Exception as e:
        logging.error("Error saving chunks to parquet: " + str(e))"""
        elif self.chunk_to_qa:
            self.worker_script += """
    train_dataset = []
    train_system_message = "{self.train_prompt}"
    path = f"{self.root}/dataset_{i}.json"
    for idx, row in df.iterrows():
        qa_pairs = llm.chunk_to_qa(
            chunk=row['chunk'],
            user_query="{self.user_query}",
            title=row['title'],
            abstract=row['abstract']
        )
        if len(qa_pairs) == 2 and isinstance(qa_pairs, dict):
            qa_pairs = [qa_pairs]
        if idx % 5 == 0 and idx != 0:
            print('Worker '+str({i})+' Progress update: '+str(idx)+'/'+str(total))
        try:
            if len(qa_pairs) != 0:
                for qa_pair in qa_pairs:
                    train_dataset.append(
                        dict(
                            messages = [
                                dict(role = "system", content = train_system_message),
                                dict(role = "user", content = qa_pair["question"]),
                                dict(role = "assistant", content = qa_pair["answer"])
                            ]
                        )
                    )
        except Exception as e:
            train_dataset.append(dict())
            logging.error('Lost 1 chunk because of LLM output')
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(train_dataset, f, indent=2)
        logging.info("train dataset completed.")
    except Exception as e:
        logging.error("train dataset failed: " + str(e))"""