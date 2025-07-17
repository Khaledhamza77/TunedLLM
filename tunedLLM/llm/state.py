from typing import TypedDict, Literal

class AgentState(TypedDict):
    model_name: str | None
    run_id: str | None
    user_query: str | None
    topic: str | None
    finetune: bool | None
    rag: bool | None
    starting_node: str | None

    job: str | None
    job_status: Literal['success', 'failure'] | None
    path_to_search_queries: str | None
    path_to_relevant_papers: str | None
    path_to_chunks: str | None
    path_to_qa_pairs: str | None
    parallel_chunking: bool | None
    parallel_qa: bool | None