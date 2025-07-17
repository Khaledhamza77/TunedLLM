import os
import json
import logging
import subprocess
import pandas as pd
from uuid import uuid4
from .agent import LLM
from ..db.logs import Logs
from .swarm import LLMSwarm
from .state import AgentState
from ..db.coredb import CoreDB
from langgraph.graph import StateGraph, END
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Graph:
    def __init__(
            self, 
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
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def onboarding(self, state: AgentState) -> AgentState:
        self.logs = Logs(self.root)
        logging.info('Starting Graph ...')
        return state
    
    def stage_routing(self, state: AgentState) -> str:
        # Check if any previous run has the same user_query
        df = self.logs.log_file
        if df is not None and not df.empty:
            match = df[df['user_query'] == state['user_query']]
            if not match.empty:
                self.logs.index = match.index[0]
                state['run_id'] = match['run_id'].iloc[0]
                for col in match.columns:
                    if pd.notna(match[col].iloc[0]) and col not in ['user_query', 'run_id']:
                        state[col] = match[col].iloc[0]
                        stage = col
                return stage
        state['run_id'] = str(uuid4()) 
        state['user_query'] = state['user_query']
        self.logs.update('run_id', state)
        self.logs.update('user_query', state)
        self.logs.save()
        os.makedirs(f"{self.root}/{state['run_id']}/data/full_texts")
        return "onboarding"

    def query_to_topic(self, state: AgentState) -> AgentState:
        state['job'] = "infer_topic_of_query"
        state['job_status'], state['topic'] = self.llm.query_to_topic(state['user_query'])
        self.logs.update('topic', state)
        return state

    def query_to_search(self, state: AgentState) -> AgentState:
        state["job"] = "query_to_search"
        state["job_status"], state["path_to_search_queries"] = self.llm.query_to_search(state['user_query'])
        self.logs.update('path_to_search_queries', state)
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
            self.logs.update('path_to_relevant_papers', state)
            logging.info("Papers and their metadata retrieved successfully.")
        except Exception as e:
            state["path_to_relevant_papers"] = ""
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
            elif state['job'] == "chunk_and_score" or state['job'] == "chunks_to_question_answer_pairs":
                state["job"] = "gpu_count"
                state["job_status"] = "success"
                state["parallel_qa"] = True
        else:
            if state['job'] == "get_papers_and_their_metadata":
                state["job"] = "gpu_count"
                state["job_status"] = "success"
                state["parallel_chunking"] = False
            elif state['job'] == "chunk_and_score" or state['job'] == "chunks_to_question_answer_pairs":
                state["job"] = "gpu_count"
                state["job_status"] = "success"
                state["parallel_qa"] = False
    
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
                        "relevance_class": self.llm.score_chunk(state["user_query"], chunk, row),
                        "title": row['title'],
                        "abstract": row['abstract']
                    },
                    ignore_index=True
                )
        try:
            proto_db.to_parquet(f"{self.root}/data/papers/chunks.parquet", index=False)
            state["path_to_chunks"] = f"{self.root}/data/papers/chunks.parquet"
            state["job_status"] = "success"
            self.logs.update('path_to_chunks', state)
            logging.info("Chunking completed.")
        except Exception as e:
            state["path_to_chunks"] = ""
            state["job_status"] = "failure"
            logging.error(f"Error saving chunks to parquet: {e}")
        return state
    
    def parallelized_chunk(self, state: AgentState) -> AgentState:
        state["job"] = "parallelized_chunk_and_score"
        swarm = LLMSwarm(
            model_name=state['model_name'],
            user_query=state['user_query'],
            root_dir=self.root,
            chunk_scoring=state['path_to_relevant_papers']
        )
        try:
            state["path_to_chunks"] = swarm.run()
            state["job_status"] = "success"
            self.logs.update('path_to_chunks', state)
            logging.info("Successfully chunked and saved chunks")
        except Exception as e:
            state["path_to_chunks"] = ""
            state["job_status"] = "failure"
            logging.error(f"Error in parallelized chunking: {e}")
        return state

    def chunks_to_qa(self, state: AgentState) -> AgentState:
        self.train_prompt = f"""You are an AI subject-matter expert which should provide expert-level answers on the followin topic:
{state['topic']}
The user will ask you a question on that topic and you will answer it fully and provide all relevant details."""
        state["job"] = "chunks_to_q/a_pairs"
        train_dataset = []
        chunks = pd.read_parquet(f'{self.root}/data/papers/chunks.parquet')
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
                            {"role": "system", "content": self.train_prompt},
                            {"role": "user", "content": qa_pair["question"]},
                            {"role": "assistant", "content": qa_pair["answer"]}
                        ]
                    }
                )
        try:
            path = f"{self.root}/data/dataset.json"
            train_dataset.to_json(path, orient="records")
            state['path_to_qa_pairs'] = path
            state['job_status'] = "success"
            self.logs.update('path_to_qa_pairs', state)
            logging.info(f'Training dataset generated successfully and saved to {path}')
        except Exception as e:
            logging.error(f'Training dataset was not saved: {e}')
            state['path_to_qa_pairs'] = ""
            state['job_status'] = 'failure'
        return state
    
    def parallelized_chunks_to_qa(self, state: AgentState) -> AgentState:
        state["job"] = "parallelized_chunks_to_q/a_pairs"
        swarm = LLMSwarm(
            model_name=state['model_name'],
            user_query=state['user_query'],
            root_dir=self.root,
            chunk_to_qa=state['path_to_chunks']
        )
        try:
            state["path_to_qa_pairs"] = swarm.run()
            state["job_status"] = "success"
            self.logs.update('path_to_qa_pairs', state)
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
        workflow.add_edge("query_to_topic", "query_to_search")
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
        workflow.add_conditional_edges(
            "parallelized_chunk",
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
        image_path = f"{self.root}/data/graph_image.png"
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
        try:
            self.llm = LLM(model_name=self.model_name, port=self.port)
        except Exception as e:
            logging.error(f"Error initializing llm: {e}")
        graph = self.build_graph()
        try:
            graph.invoke(state)
            logging.info("Graph execution completed successfully.")
        except Exception as e:
            logging.error(f"Error during graph execution: {e}")
            raise RuntimeError(f"Graph execution failed: {e}")