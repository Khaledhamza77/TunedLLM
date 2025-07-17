# TunedLLM
Agentic AI system for making your LLM tuned with latest research papers relevant to any query

# Setup

Get API key from CORE database: https://core.ac.uk/services/api#what-is-included

Follow the instructions step by step and then save the api key string in .txt file called apikey.txt in your working directory.
```bash
    docker pull ollama/ollama
```
```bash
    docker run -d --gpus '"device=0"' -v ollama_gpu0:/root/.ollama -p 11434:11434 --name ollama_gpu0 ollama/ollama
    docker run -d --gpus '"device=1"' -v ollama_gpu1:/root/.ollama -p 11435:11434 --name ollama_gpu1 ollama/ollama
    docker run -d --gpus '"device=2"' -v ollama_gpu2:/root/.ollama -p 11436:11434 --name ollama_gpu2 ollama/ollama
    docker run -d --gpus '"device=3"' -v ollama_gpu3:/root/.ollama -p 11437:11434 --name ollama_gpu3 ollama/ollama
```

```bash
    docker exec -it ollama_gpu0 ollama pull gemma3:1b
    docker exec -it ollama_gpu1 ollama pull gemma3:1b
    docker exec -it ollama_gpu2 ollama pull gemma3:1b
    docker exec -it ollama_gpu3 ollama pull gemma3:1b
```
