import os
import json
import logging
import pandas as pd
from uuid import uuid4
from .agent import Gemma
from ..coredb import CoreDB
from .state import AgentState
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Graph:
    def __init__(self, model_name: str = "gemma3:1b", port: str = "11434"):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.root = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
        os.makedirs(f"{self.root}/data", exist_ok=True)
        os.makedirs(f"{self.root}/data/papers", exist_ok=True)
        os.makedirs(f"{self.root}/data/papers/full_texts", exist_ok=True)
        self.gemma = Gemma(model_name=model_name, port=port)
    
    def query_to_search(self, state: AgentState, query: str) -> AgentState:
        state["run_id"] = str(uuid4())
        state["user_query"] = query
        state["job"] = "query_to_search"
        state["action_status"], state["path_to_search_queries"] = self.gemma.query_to_search(query)
        return state

    def get_papers(self, state: AgentState) -> AgentState:
        core = CoreDB(self.root)
        ceilings = [500, 300, 200, 100]
        state["job"] = "get_papers_and_their_metadata"
        with open(state["path_to_search_queries"], 'r', encoding='utf-8') as f:
            data = json.load(f)
        search_queries_dict = data.get("search_queries")
        sorted_keys = sorted(search_queries_dict.keys(), key=int)
        search_queries_list = [search_queries_dict[key] for key in sorted_keys]
        try:
            for i, query in enumerate(search_queries_list):
                core.scroll(query, ceiling=ceilings[i], i=i)
            core.cocnat_metadata()
            state["path_to_relevant_papers"] = f"{self.root}/data/papers"
            state["job_status"] = "success"
            logging.info("Papers and their metadata retrieved successfully.")
        except Exception as e:
            state["path_to_relevant_papers"] = ""
            state["job_status"] = "failure"
            logging.error(f"Error retrieving papers: {e}")
        return state
    
    def chunk(self, state: AgentState) -> AgentState:
        state["job"] = "chunk_papers"
        proto_db = pd.DataFrame()
        metadata = pd.read_parquet(f"{state['path_to_relevant_papers']}/metadata.parquet")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        for _, row in metadata.iterrows():
            with open(f"{self.root}/data/papers/full_texts/{row['id']}.txt", 'r', encoding='utf-8') as f:
                full_text = f.read()
            chunks = text_splitter.split_text(full_text)
            for chk_idx, chunk in enumerate(chunks):
                proto_db = proto_db.append(
                    {
                        "chunk_id": f"{row['id']}_{chk_idx}",
                        "id": row["id"],
                        "chunk": chunk,
                        "relevance_class": self.gemma.score_chunk(state["user_query"], row)
                    },
                    ignore_index=True
                )
        state["job_status"] = "success"
        logging.info("Chunking completed.")
        return state
    
    def parallelized_chunk(self, state: AgentState) -> AgentState:
        return state

    def chunks_to_qa(self, state: AgentState) -> AgentState:
        state["job"] = "chunks_to_question_answer_pairs"
        return state
    
    def parallelized_chunks_to_qa(self, state: AgentState) -> AgentState:
        return state