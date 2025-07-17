import os
from .llm.graph import Graph


class app:
    def __init__(self):
        self.root = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/') + "/tunedLLM"
        os.makedirs(self.root, exist_ok=True)