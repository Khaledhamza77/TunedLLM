import os
import json
import logging
import subprocess
import numpy as np
import pandas as pd
from uuid import uuid4
from langgraph.graph import StateGraph, END
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .agent import LLM
from ..db.logs import Logs
from .swarm import LLMSwarm
from .state import AgentState
from ..ft.trainer import Tuner
from ..db.coredb import CoreDB


class Graph:
    def __init__(
            self,
            diversify: bool,
            total_docs: int,
            user_query: str,
            root_dir: str, 
            model_name: str = "gemma3:1b", 
            port: str = "11434",
            finetune: bool = True,
            rag: bool = False
        ):
        self.user_query = user_query
        self.root = root_dir
        self.model_name = model_name
        self.port = port
        self.finetune = finetune
        self.rag = rag
        self.total_docs = total_docs
        self.diversify = diversify
        logging.getLogger('ollama').setLevel(logging.WARNING)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def get_state_job(self, stage: str = None):
        if stage == 'topic':
            return 'infer_topic_of_query', 'success'
        elif stage == 'path_to_search_queries':
            return 'query_to_search', 'success'
        elif stage == 'path_to_relevant_papers':
            return 'get_papers_and_their_metadata', 'success'
        elif stage == 'path_to_chunks':
            return 'chunk_and_score', 'success'
        elif stage == 'path_to_qa_pairs':
            return 'chunks_to_q/a_pairs', 'success'
    
    def onboarding(self, state: AgentState) -> AgentState:
        self.logs = Logs(self.root)
        df = self.logs.log_file
        matching_rows = df[df['user_query'] == state['user_query']]
        if not matching_rows.empty:
            target_row = matching_rows.iloc[0]
            state['run_id'] = target_row['run_id']
            state['starting_node'] = 'onboarding'
            for col in df.columns:
                if col not in ['user_query', 'run_id']:
                    cell_value = target_row[col]
                    if not pd.isna(cell_value) and cell_value.strip() != "":
                        state['starting_node'] = col
                        state['job'], state['job_status'] = self.get_state_job(col)
                        state[col] = cell_value
        else:
            state['run_id'] = str(uuid4())
            state['starting_node'] = 'onboarding'
            self.logs.update('run_id', state)
            self.logs.update('user_query', state)
            self.logs.save()
            os.makedirs(f"{self.root}/{state['run_id']}/data/full_texts")
        logging.info('Starting Graph ...')
        return state
    
    def stage_routing(self, state: AgentState) -> str:
        return state['starting_node']

    def query_to_topic(self, state: AgentState) -> AgentState:
        state['job'] = "infer_topic_of_query"
        state['job_status'], state['topic'] = self.llm.query_to_topic(state['user_query'])
        self.logs.update('topic', state)
        self.logs.save()
        return state

    def diversify_query(self, state: AgentState) -> str:
        if state['diversify']:
            return 'augment_search_query'
        else:
            return 'topic_as_search_query'
    
    def topic_as_query(self, state: AgentState) -> AgentState:
        state["job"] = "query_to_search"
        try:
            sq = {
                "search_queries": {
                    "0": state['topic']
                }
            }
            path_to_search_queries = f"{self.root}/{state['run_id']}/data/search_queries.json"
            with open(path_to_search_queries, 'w') as f:
                json.dump(sq, f, indent=4)
            state["job_status"] = "success"
            state["path_to_search_queries"] = path_to_search_queries
            self.logs.update('path_to_search_queries', state)
            self.logs.save()
            logging.info("Saved topic as search query")
            return state
        except Exception as e:
            logging.error(f"Failed to set topic as search query: {e}")
            state['job_status'] = 'failure'
            return state

    def query_to_search(self, state: AgentState) -> AgentState:
        state["job"] = "query_to_search"
        state["job_status"], state["path_to_search_queries"] = self.llm.query_to_search(state)
        self.logs.update('path_to_search_queries', state)
        self.logs.save()
        return state

    def get_papers(self, state: AgentState) -> AgentState:
        core = CoreDB(f"{self.root}/{state['run_id']}")
        if state['diversify']:
            ceilings = [
                int(self.total_docs/2),
                int(self.total_docs/4),
                int(self.total_docs/8),
                int(self.total_docs/16)]
        else:
            ceilings = [self.total_docs]
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
            state["path_to_relevant_papers"] = f"{self.root}/{state['run_id']}/data/metadata.parquet"
            state["job_status"] = "success"
            self.logs.update('path_to_relevant_papers', state)
            self.logs.save()
            logging.info("Papers and their metadata retrieved successfully.")
        except Exception as e:
            state["path_to_relevant_papers"] = np.nan
            state["job_status"] = "failure"
            logging.error(f"Error retrieving papers: {e}")
        return state
    
    def gpu_info(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                logging.error(f"nvidia-smi error: {result.stderr.strip()}")
                raise RuntimeError('')

            memory_limits = [int(x.strip()) for x in result.stdout.strip().split("\n")]
            gpu_count = len(memory_limits)
            
            logging.info(f"Found {gpu_count} GPU(s); {memory_limits[0]}MB (each).")
            return gpu_count

        except FileNotFoundError:
            logging.error("nvidia-smi not found. Ensure NVIDIA drivers are installed.")
            raise RuntimeError('')
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            raise RuntimeError('')
    
    def parallelize_node(self, state: AgentState) -> AgentState:
        try:
            count = self.gpu_info()
        except RuntimeError as e:
            logging.error(f"Error checking GPU info: {e}")

        if count == 4:
            if state['job'] == "get_papers_and_their_metadata":
                state["job"] = "gpu_count"
                state["job_status"] = "success"
                state["parallel_chunking"] = True
            elif state['job'] == "chunk_and_score" or state['job'] == "parallelized_chunk_and_score":
                state["job"] = "gpu_count"
                state["job_status"] = "success"
                state["parallel_qa"] = True
        else:
            if state['job'] == "get_papers_and_their_metadata":
                state["job"] = "gpu_count"
                state["job_status"] = "success"
                state["parallel_chunking"] = False
            elif state['job'] == "chunk_and_score" or state['job'] == "parallelized_chunk_and_score":
                state["job"] = "gpu_count"
                state["job_status"] = "success"
                state["parallel_qa"] = False
        return state
    
    def chunk(self, state: AgentState) -> AgentState:
        state["job"] = "chunk_and_score"
        proto_db = pd.DataFrame()
        metadata = pd.read_parquet(state['path_to_relevant_papers'])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        for _, row in metadata.iterrows():
            with open(f"{self.root}/{state['run_id']}/data/full_texts/{row['id']}.txt", 'r', encoding='utf-8') as f:
                full_text = f.read()
            chunks = text_splitter.split_text(full_text)
            for chk_idx, chunk in enumerate(chunks):
                proto_db = proto_db.append(
                    {
                        "chunk_id": f"{row['id']}_{chk_idx}",
                        "id": row["id"],
                        "chunk": chunk,
                        "relevance_class": self.llm.score_chunk(state["user_query"], chunk, row),
                        "title": row['title'],
                        "abstract": row['abstract']
                    },
                    ignore_index=True
                )
        try:
            proto_db.to_parquet(f"{self.root}/{state['run_id']}/data/chunks.parquet", index=False)
            state["path_to_chunks"] = f"{self.root}/{state['run_id']}/data/chunks.parquet"
            state["job_status"] = "success"
            self.logs.update('path_to_chunks', state)
            self.logs.save()
            logging.info("Chunking completed.")
        except Exception as e:
            state["path_to_chunks"] = np.nan
            state["job_status"] = "failure"
            logging.error(f"Error saving chunks to parquet: {e}")
        return state
    
    def parallelized_chunk(self, state: AgentState) -> AgentState:
        state["job"] = "parallelized_chunk_and_score"
        swarm = LLMSwarm(
            model_name=state['model_name'],
            user_query=state['user_query'],
            root_dir=f"{self.root}/{state['run_id']}/data",
            chunk_scoring=state['path_to_relevant_papers']
        )
        try:
            state["path_to_chunks"] = swarm.run()
            state["job_status"] = "success"
            self.logs.update('path_to_chunks', state)
            self.logs.save()
            logging.info("Successfully chunked and saved chunks")
        except Exception as e:
            state["path_to_chunks"] = ""
            state["job_status"] = "failure"
            logging.error(f"Error in parallelized chunking: {e}")
        return state

    def chunks_to_qa(self, state: AgentState) -> AgentState:
        train_prompt = f"You are an AI subject-matter expert which should provide expert-level answers on the followin topic: '{state['topic']}'. The user will ask you a question on that topic and you will answer it fully and provide all relevant details."
        state["job"] = "chunks_to_q/a_pairs"
        train_dataset = []
        chunks = pd.read_parquet(state["path_to_chunks"])
        chunks = chunks[chunks["relevance_class"] == "a"]
        for _, row in chunks.iterrows():
            qa_pairs = self.llm.chunk_to_qa(
                chunk=row['chunk'],
                user_query=state['user_query'],
                title=row['title'],
                abstract=row['abstract']
            )
            for qa_pair in qa_pairs:
                train_dataset.append(
                    {
                        "messages": [
                            {"role": "system", "content": train_prompt},
                            {"role": "user", "content": qa_pair["question"]},
                            {"role": "assistant", "content": qa_pair["answer"]}
                        ]
                    }
                )
        try:
            path = f"{self.root}/{state['run_id']}/data/dataset.json"
            train_dataset.to_json(path, orient="records")
            state['path_to_qa_pairs'] = path
            state['job_status'] = "success"
            self.logs.update('path_to_qa_pairs', state)
            self.logs.save()
            logging.info(f'Training dataset generated successfully and saved to {path}')
        except Exception as e:
            logging.error(f'Training dataset was not saved: {e}')
            state['path_to_qa_pairs'] = ""
            state['job_status'] = 'failure'
        return state
    
    def parallelized_chunks_to_qa(self, state: AgentState) -> AgentState:
        state["job"] = "parallelized_chunks_to_q/a_pairs"
        train_prompt = f"You are an AI subject-matter expert which should provide expert-level answers on the followin topic: '{state['topic']}'. The user will ask you a question on that topic and you will answer it fully and provide all relevant details."
        swarm = LLMSwarm(
            model_name=state['model_name'],
            user_query=state['user_query'],
            root_dir=f"{self.root}/{state['run_id']}/data",
            chunk_to_qa=state['path_to_chunks'],
            train_prompt=train_prompt
        )
        try:
            state["path_to_qa_pairs"] = swarm.run(qa=True)
            state["job_status"] = "success"
            self.logs.update('path_to_qa_pairs', state)
            self.logs.save()
        except Exception as e:
            state["path_to_qa_pairs"] = ""
            state["job_status"] = "failure"
            logging.error(f"Error in parallelized chunking: {e}")
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
        
    def finetune(self, state: AgentState) -> AgentState:
        return state
    
    def build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node(
            "onboarding",
            self.onboarding
        )
        workflow.add_node(
            "query_to_topic",
            self.query_to_topic
        )
        workflow.add_node(
            "topic_to_search",
            self.topic_as_query
        )
        workflow.add_node(
            "query_to_search",
            self.query_to_search,
        )
        workflow.add_node(
            "get_papers",
            self.get_papers,
        )
        workflow.add_node(
            "check_gpu_infrastructure_1",
            self.parallelize_node,
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
            "check_gpu_infrastructure_2",
            self.parallelize_node,
        )
        workflow.add_node(
            "chunks_to_qa",
            self.chunks_to_qa,
        )
        workflow.add_node(
            "parallelized_chunks_to_qa",
            self.parallelized_chunks_to_qa,
        )
        workflow.set_entry_point("onboarding")
        workflow.add_conditional_edges(
            "onboarding",
            self.stage_routing,
            {
                "onboarding": "query_to_topic",
                "topic": "query_to_search",
                "path_to_search_queries": "get_papers",
                "path_to_relevant_papers": "check_gpu_infrastructure_1",
                "path_to_chunks": "check_gpu_infrastructure_2",
                "path_to_qa_pairs": END
            }
        )
        workflow.add_conditional_edges(
            "query_to_topic",
            self.diversify_query,
            {
                "augment_search_query": "query_to_search",
                "topic_as_search_query": "topic_to_search"
            }
        )
        workflow.add_edge("topic_to_search", "get_papers")
        workflow.add_edge("query_to_search", "get_papers")
        workflow.add_edge("get_papers", "check_gpu_infrastructure_1")
        workflow.add_conditional_edges(
            "check_gpu_infrastructure_1",
            self.chunking_routing,
            {
                "single": "chunk",
                "parallel": "parallelized_chunk"
            }
        )
        workflow.add_edge("chunk", "check_gpu_infrastructure_2")
        workflow.add_edge("parallelized_chunk", "check_gpu_infrastructure_2")
        workflow.add_conditional_edges(
            "check_gpu_infrastructure_2",
            self.qa_pairs_routing,
            {
                "single": "chunks_to_qa",
                "parallel": "parallelized_chunks_to_qa"
            }
        )
        workflow.add_edge("chunks_to_qa", END)
        workflow.add_edge("parallelized_chunks_to_qa", END)
        try:
            graph = workflow.compile()
            logging.info("Graph compiled successfully.")
        except Exception as e:
            logging.error(f"Error compiling graph: {e}")
            raise RuntimeError(f"Graph compilation failed: {e}")
        image_data = graph.get_graph(xray=True).draw_mermaid_png()
        image_path = f"{self.root}/graph_image.png"
        with open(image_path, "wb") as f:
            f.write(image_data)
        logging.info("Graph image generated successfully.")
        return graph
    
    def run(self):
        state = AgentState()
        state["model_name"] = self.model_name
        state["user_query"] = self.user_query
        state["finetune"] = self.finetune
        state["rag"] = self.rag
        state["diversify"] = self.diversify
        try:
            self.llm = LLM(root_dir=self.root, model_name=self.model_name, port=self.port)
        except Exception as e:
            logging.error(f"Error initializing llm: {e}")
        graph = self.build_graph()
        try:
            graph.invoke(state)
            logging.info("Graph execution completed successfully.")
        except Exception as e:
            logging.error(f"Error during graph execution: {e}")
            raise RuntimeError(f"Graph execution failed: {e}")