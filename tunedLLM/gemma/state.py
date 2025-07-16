from typing import TypedDict, Literal

class AgentState(TypedDict):
    run_id: str | None
    user_query: str | None
    job: str | None
    job_status: Literal['success', 'failure'] | None
    path_to_search_queries: str | None
    path_to_relevant_papers: str | None
    path_to_chunks: str | None
