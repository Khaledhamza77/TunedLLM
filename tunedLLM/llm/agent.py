import json
import ollama
import logging

class LLM:
    def __init__(self, root_dir: str = None, model_name: str = "gemma3:1b", port: str = "11434"):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.client = ollama.Client(host=f"http://localhost:{port}")
        self.root = root_dir
        self.model_name = model_name
        logging.info(f"LLM client initialized with model {self.model_name} on port {port}")
        self.query_to_topic_system_message = """You are an AI agent which helps users find the best topic which best encompasses their query.
You will be provided with a query and you should tell the user what topic best describes and encompasses their query. The topic should be general and exhaustive.
You should only output the topic in the following structure:
{
    'topic': 'topic you generated'
}
You should not generate anything else. The topic should be general enough to contain all information that might be needed to answer user query."""
        self.query_to_search_system_message = """You are an AI agent that helps users transform their queries into 4 search queries which can be used to search the CORE database which is a comprehensive bibliographic database of the world's scholarly literature and it is the world's largest collection of open access research papers
You will be given a query and you will return 4 search queries that can be used to search the CORE database
You will return the search queries in a json format with the following structure:
{
    "search_queries": {
        "0": "search query 1",
        "1": "search query 2",
        "2": "search query 3",
        "3": "search query 4"
    }
}
You should not return any other text or explanation, just the json object with the search queries.
You should order them based on their relevance to the original query, with the most relevant query first."""

        self.score_chunk_system_message = """You are an AI agent which identifies the relevance of a given chunk to both the user query and the title and abstract of a paper from which the chunk is retrieved.
You will be given a chunk, a user query, a paper title and a paper abstract.
You will return a relevance class from the following list:
- 'a': The chunk is highly relevant to the user query, title and abstract.
- 'b': The chunk is relevant to the user query, title and abstract.
- 'c': The chunk is somewhat relevant to the user query, title and abstract.
- 'd': The chunk is not relevant to the user query, title and abstract.
Description of a class 'a' chunk: It contains key information that directly addresses the user query and is closely related to the title and abstract of the paper.
Description of a class 'b' chunk: It contains important information that is relevant to the user query and is related to the title and abstract of the paper.
Description of a class 'c' chunk: It contains some information that is somewhat relevant to the user query and is loosely related to the title and abstract of the paper.
Description of a class 'd' chunk: It doesn't contain any infromation relevant to user query and is not discussing the title and abstract of the paper. It could be a chunk of names, citations, dates, undescribed data, etc.
You will return the relevance class in a json format with the following structure:
{
    "relevance_class": "a" | "b" | "c" | "d"
}"""
        self.chunk_to_qa_system_message = """You are an AI agent which transforms a chunk of text into question and answer pairs. You will be given a chunk of text and you will return a set of question and answer pairs that can be used to fine tune a language model.
For every relevant piece of informaion in the chunk, you will create a question and an answer. The question should be a clear and concise question that can be answered by the chunk. The answer should be a clear and concise answer that is directly related to the question.
You will also receive a user query, a paper title and a paper abstract. You will use these to create the question and answer pairs such that they are relevant to the user query, title, and cover all information in the chunk.
You should return 1 or more pairs based on the amount of relevant information in the chunk. If there is no information to be found in the chunk you can return 0 pairs.
You should not return any other text or explanation, just the json object with the question and answer pairs.
You should not ask general questions like 'what the paper is about?' or any similar question.
Your questions should be directly discussing the topic and not general or related to authors or general information regarding chunk/paper.
You will return the question and answer pairs in a json format strictly with the following structure:
In case of 0 pairs:
{
    "pairs": {
    }
}

In case of 1 pair:
{
    "pairs": {
        {
            "question": "question 1 text",
            "answer": "answer 1 text"
        }
    }
}

In case of 2 pairs or more:
{
    "pairs": {
        {
            "question": "question 1 text",
            "answer": "answer 1 text"
        },
        {
            "question": "question 2 text",
            "answer": "answer 2 text"
        },
        ...
    }
}"""    
    def query_to_search(self, state):
        messages = [
            {'role': 'system', 'content': self.query_to_search_system_message},
            {'role': 'user', 'content': state['user_query']}
        ]
        resp = self.client.chat(
            model=self.model_name,
            messages=messages,
            format='json',
            options={
                'temperature': 0.1
            }
        )
        try:
            response = json.loads(resp['message']['content'])
            path_to_search_queries = f"{self.root}/{state['run_id']}/data/search_queries.json"
            with open(path_to_search_queries, 'w') as f:
                json.dump(response, f, indent=4)
            action_status = "success"
            logging.info(f"Search queries saved to search_queries.json: {response}")
        except json.JSONDecodeError as e:
            path_to_search_queries = ""
            action_status = "failure"
            logging.error(f"Error decoding JSON response from Ollama: {e}")
        
        return action_status, path_to_search_queries
    
    def query_to_topic(self, query: str):
        messages = [
            {'role': 'system', 'content': self.query_to_topic_system_message},
            {'role': 'user', 'content': query}
        ]
        try:
            resp = self.client.chat(
                model=self.model_name,
                messages=messages,
                format='json',
                options={
                    'temperature': 0.1
                }
            )
            response = json.loads(resp['message']['content'])
            topic = response['topic']
            logging.info(f"Generated topic for query successfully: {topic}")
            return "sucess", topic
        except Exception as e:
            logging.error(f'Could not generate topic: {e}')
            return "failure", ""
    
    def score_chunk(self, user_query, chunk, doc):
        user_message_content = f"""Chunk: {chunk}
User Query: {user_query}
Paper Title: {doc['title']}
Paper Abstract: {doc['abstract']}"""
        messages = [
            {'role': 'system', 'content': self.score_chunk_system_message},
            {'role': 'user', 'content': user_message_content}
        ]
        max_retries = 5
        for attempt in range(max_retries):
            resp = self.client.chat(
                model=self.model_name,
                messages=messages,
                format='json',
                options={
                    'temperature': 0.1
                }
            )
            try:
                response = json.loads(resp['message']['content'])
                relevance_class = response['relevance_class']
                return relevance_class
            except (KeyError, json.JSONDecodeError) as e:
                logging.warning(f"Attempt {attempt+1}/{max_retries}: Error in response format from Ollama: {e}")
        logging.error("Failed to get valid relevance_class after multiple attempts.")
        return ""

    def chunk_to_qa(self, chunk: str, user_query: str, title: str, abstract: str):
        user_message_content = f"""User Query: {user_query}
Paper Title: {title}
Abstract: {abstract}
Chunk: {chunk}"""
        messages = [
            {'role': 'system', 'content': self.chunk_to_qa_system_message},
            {'role': 'user', 'content': user_message_content}
        ]
        resp = self.client.chat(
            model=self.model_name,
            messages=messages,
            format='json',
            options={
                'temperature': 0.1
            }
        )
        try:
            qa_pairs = json.loads(resp['message']['content'])
            return qa_pairs['pairs']
        except Exception as e:
            logging.error(f"Error in response format from Ollama: {e}")
            return {}