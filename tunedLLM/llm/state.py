from typing import TypedDict, Literal

class AgentState(TypedDict):
    model_name: str | None
    run_id: str | None
    user_query: str | None
    diversify: bool | None
    topic: str | None
    finetune: bool | None
    rag: bool | None
    starting_node: str | None
    early_exit_at: Literal['get_papers_and_their_metadata', 'chunk_and_score', 'parallelized_chunk_and_score', 'after_chunks_to_q/a_pairstuning', 'parallelized_chunks_to_q/a_pairs', 'finetuning_model'] | None

    job: Literal['infer_topic_of_query', 'query_to_search', 'get_papers_and_their_metadata', 'chunk_and_score', 'parallelized_chunk_and_score', 'chunks_to_q/a_pairs', 'parallelized_chunks_to_q/a_pairs', 'setting_up_tuning_script', 'finetuning_model'] | None
    job_status: Literal['success', 'failure'] | None
    path_to_search_queries: str | None
    path_to_relevant_papers: str | None
    path_to_chunks: str | None
    path_to_qa_pairs: str | None
    parallel_chunking: bool | None
    parallel_qa: bool | None

    tuning_script_path: str | None
    standalone_model_path: str | None
    benchmark_results_path: str | None