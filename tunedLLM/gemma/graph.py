import os
import glob
import json
import logging
import pandas as pd
from uuid import uuid4
from .agent import Gemma
from ..coredb import CoreDB
from .state import AgentState
from .pargemma import ParallelGemma
from langgraph.graph import StateGraph, END
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Graph:
    def __init__(self, user_query: str, model_name: str = "gemma3:1b", port: str = "11434"):
        self.user_query = user_query
        self.model_name = model_name
        self.port = port
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.root = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
        os.makedirs(f"{self.root}/data", exist_ok=True)
        os.makedirs(f"{self.root}/data/papers", exist_ok=True)
        os.makedirs(f"{self.root}/data/papers/full_texts", exist_ok=True)
    
    def query_to_search(self, state: AgentState) -> AgentState:
        state["model_name"] = self.model_name
        state["run_id"] = str(uuid4())
        state["user_query"] = self.user_query
        state["job"] = "query_to_search"
        try:
            self.gemma = Gemma(model_name=self.model_name, port=self.port)
        except Exception as e:
            logging.error(f"Error initializing Gemma: {e}")
        state["action_status"], state["path_to_search_queries"] = self.gemma.query_to_search(self.user_query)
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
            core.concat_metadata()
            state["path_to_relevant_papers"] = f"{self.root}/data/papers"
            state["job_status"] = "success"
            logging.info("Papers and their metadata retrieved successfully.")
        except Exception as e:
            state["path_to_relevant_papers"] = ""
            state["job_status"] = "failure"
            logging.error(f"Error retrieving papers: {e}")
        return state
    
    def chunk(self, state: AgentState) -> AgentState:
        state["job"] = "chunk_and_score"
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
                        "relevance_class": self.gemma.score_chunk(state["user_query"], chunk, row)
                    },
                    ignore_index=True
                )
        try:
            proto_db.to_parquet(f"{self.root}/data/papers/chunks.parquet", index=False)
            state["path_to_chunks"] = f"{self.root}/data/papers/chunks.parquet"
            state["job_status"] = "success"
            logging.info("Chunking completed.")
        except Exception as e:
            state["path_to_chunks"] = ""
            state["job_status"] = "failure"
            logging.error(f"Error saving chunks to parquet: {e}")
        return state
    
    def parallelized_chunk(self, state: AgentState) -> AgentState:
        state["job"] = "parallelized_chunk_and_score"
        pargemma = ParallelGemma(
            model_name=state['model_name'],
            user_query=state['user_query'],
            root_dir=self.root,
            chunk_scoring=state['path_to_relevant_papers']
        )
        try:
            state["path_to_chunks"] = pargemma.run()
            state["job_status"] = "success"
        except Exception as e:
            state["path_to_chunks"] = ""
            state["job_status"] = "failure"
            logging.error(f"Error in parallelized chunking: {e}")
        return state

    def chunks_to_qa(self, state: AgentState) -> AgentState:
        state["job"] = "chunks_to_question_answer_pairs"
        return state
    
    def parallelized_chunks_to_qa(self, state: AgentState) -> AgentState:
        return state
    
    def chunking_routing(self, state: AgentState) -> str:
        if state['parallel_chunking']:
            return "parallel"
        else:
            return "single"

    def qa_pairs_routing(self, state: AgentState) -> str:
        if state['parallel_qa']:
            return "parallel"
        else:
            return "single"
    
    def build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node(
            "query_to_search",
            self.query_to_search,
        )
        workflow.add_node(
            "get_papers",
            self.get_papers,
        )
        workflow.add_node(
            "chunk",
            self.chunk,
        )
        workflow.add_node(
            "parallelized_chunk",
            self.parallelized_chunk,
        )
        workflow.add_node(
            "chunks_to_qa",
            self.chunks_to_qa,
        )
        workflow.add_node(
            "parallelized_chunks_to_qa",
            self.parallelized_chunks_to_qa,
        )

        workflow.set_start("query_to_search")
        workflow.add_edge("query_to_search", "get_papers")
        workflow.add_condditional_edge(
            "get_papers",
            self.chunking_routing,
            {
                "single": "chunk",
                "parallel": "parallelized_chunk"
            }
        )
        workflow.add_conditional_edge(
            "chunk",
            self.qa_pairs_routing,
            {
                "single": "chunks_to_qa",
                "parallel": "parallelized_chunks_to_qa"
            }
        )
        workflow.add_conditional_edge(
            "parallelized_chunk",
            self.qa_pairs_routing,
            {
                "single": "chunks_to_qa",
                "parallel": "parallelized_chunks_to_qa"
            }
        )
        workflow.add_edge("chunks_to_qa", END)
        workflow.add_edge("parallelized_chunks_to_qa", END)
        graph = workflow.compile()
        return graph