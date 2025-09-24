# TunedLLM

## Overview
The TunedLLM package is designed to create an automated parallelized agentic workflow which is able to take a user's query and transform it into a tuned LLM either through a retrieval system or through straightforwardly fine-tuning the chosen model. This package relies on Ollama, LangChain, LangGraph, and more. The setup would be the most complex part, but I will do my best to provide the best setup strategy.

## Features
This package is able to perform the following:
1. **Query Augmentation**: Infer the topic of the query and use the query and topic to create a search query for the CORE database to get relevant research papers. There is an optional step of creating multiple search queries regarding to cover more ground to search the database with.
2. **Research Paper Retrieval**: CORE database is queried with the search query(s) prepared in the previous node. This stage has retry logic and error handling to avoid faliure.
3. **Papers to Scored Chunks**: The response retrieved from CORE is parsed and chunked where every chunk is given a "coherence score" which is provided by an agent.
    - The swarm.py is able to create a number of workers all doing the work at the same time based on the setup and infrastructure check. So this step can be parallelized.
4. **Chunks to Q/A pairs**: The top-scoring chunks are transformed into Q/A pairs making them ready for finetuning an LLM (TODO: or are transformed into chunks and their embeddings in an index for RAG systems)
    - Similar to the step above, swarm.py is able to create a number of LLM workers which execute this task in parallel.
6. **Fine-Tuning**: Using transformers and PyTorch an LLM is fine-tuned (PEFT).
5. **Evaluation**: The finetuned model or the RAG system are evaluated and results are saved (TODO)

![Alt text](./graph.png)

## Installation
To get started with TunedLLM, you can install it directly from this GitHub repository using pip.
1. Ensure Git is installed on your system
2. Install TunedLLM using the following command:
```bash
pip install git+https://github.com/Khaledhamza77/TunedLLM
```

## Setup
The setup for this package has three parts:
1. Access to CORE database
2. Setting up Ollama and enabling parallel workers
3. Setting up HF and PyTorch for GPU-accelerated and parallelized model finetuning

### CORE Database
Get the API key from CORE database: https://core.ac.uk/services/api#what-is-included
Follow the instructions step by step and then save the api key string in .txt file called apikey.txt in your working directory.

### Ollama
This setup you will need Docker, so make sure docker is installed and ready to work with on your system.

The first step would be pulling the Ollama docker image to your local system and creating a shared volume such that there will be no need to replicate model files and other storage dependencies for all ollama workers.
```bash
SHARED_OLLAMA_VOLUME="ollama_models_shared"
echo "Creating shared Ollama model volume: $SHARED_OLLAMA_VOLUME"
docker volume create $SHARED_OLLAMA_VOLUME
docker pull ollama/ollama
```
The following step needs your personal judgement so make sure this is ready in the same way Git and Docker was. Based on the number of GPUs and the GPU memory each ollama worker uses while running a model, you will choose the CONTAINERS_PER_GPU parameter. I used Gemma 1B and it uses around 1GB of GPU memory when I am using it on a T4 NVIDIA GPU. So I can choose to have 10 workers, for example, per GPU (I had 4), and that would leave 5GB buffer on my GPU since this GPU had 15GB of memory. So in total I would have 40 ollama workers ready for me to use in the parallelizable processes of 1) chunking and scoring or 2) generating q/a pairs. The following code will create those workers and set the appropriate ports which will be called using ollama later through the package.
```bash
SHARED_OLLAMA_VOLUME="ollama_models_shared"
BASE_PORT=11434
NUM_GPUS=4
CONTAINERS_PER_GPU=10
TOTAL_CONTAINERS=$((NUM_GPUS * CONTAINERS_PER_GPU))
for i in $(seq 0 $((TOTAL_CONTAINERS - 1))); do
    GPU_DEVICE=$((i / CONTAINERS_PER_GPU))
    HOST_PORT=$((BASE_PORT + i))
    CONTAINER_NAME="ollama_worker_${i}"

    echo "Starting container ${CONTAINER_NAME} on GPU ${GPU_DEVICE} with host port ${HOST_PORT}..."
    docker run -d \
        --gpus "device=${GPU_DEVICE}" \
        -v ${SHARED_OLLAMA_VOLUME}:/root/.ollama \
        -p ${HOST_PORT}:11434 \
        --name ${CONTAINER_NAME} \
        ollama/ollama
    if [ $? -ne 0 ]; then
        echo "Error starting container ${CONTAINER_NAME}. Continuing with others."
    fi
done
```
Finally, pull your desired model using any worker and (thanks to the shared volume) all other workers will have access to that model and can run it on the GPU assigned to the worker.
```bash
docker exec -it ollama_worker_0 ollama pull gemma3:1b
```

### HuggingFace and PyTorch
This could either be a nightmare or a walk in the park. You can use AWS Deep Learning Containers with a PyTorch and CUDA setup ready to run this package from within, otherwise you'd have to download PyTorch, CUDA, and all their dependencies suitable for you GPU. For HuggingFace, you can generate a token from here: https://huggingface.co/docs/hub/security-tokens and use it below:
```bash
from huggingface_hub import login
 
login(token=YOUR_HF_TOKEN, add_to_git_credential=True)
```
Now you're READY!

## Demo