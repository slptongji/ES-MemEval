# ES-MemEval

This repository contains the dataset and code for our paper: "ES-MemEval: Benchmarking Conversational Agents on Personalized Long-Term Emotional Support".

## Enviroment Setup

### Dependency Installation

To execute the code in this repository, please install the required dependencies using Anaconda with the following commands:

```sh
conda create -n ES_MemEval python=3.13.5
conda activate ES_MemEval
pip install -r requirements.txt
```

The experiments were conducted on the following environments and have been verified to run successfully on various configurations:

- Operating System: Ubuntu 18.04.5 LTS (also tested on Ubuntu 24.04.3 LTS)

- GPU: NVIDIA TITAN RTX / GeForce RTX 3080 / A100

- CUDA Version: 12.4 (also verified with CUDA 13.0)

### Python Configuration

The `PYTHONPATH` environment variable should be configured to point to the `src` directory to ensure correct module imports :

```sh
export PYTHONPATH="./src"    # Should be replaced with an absolute path, unless the working directory is fixed to the repository root. (In VSCode, this will be automatically configured.)
```

### Project Configuration

All project-level configurations are defined in `src/exe/common_configurations.py`, which specifies key parameters such as API endpoints for large language models (LLMs), data and output paths, and other global settings.

#### LLM Configuation

By default,  the OpenAI API key is loaded from
 `./secrets/open_ai_api_key.txt` (a relative path).
 To modify this behavior, locate and update the following expression in `common_configurations.py`:

```python
_csfile.read_all_text("./secrets/open_ai_api_key.txt")
```

Likewise, the default configuration defines other LLM endpoints as locally hosted on `_vllm_host`, exposed through OpenAI-compatible APIs on ports `8911`–`8913`. 

These parameters can be modified in `common_configurations.py` if alternative hosts or ports are required. 

To deploy equivalent services locally, the models can be launched using **vLLM** as illustrated below:

```sh
# Create a vLLM environment
conda create -n ES_MemEval_vLLM python=3.12
conda activate ES_MemEval_vLLM
pip install uv
uv pip install vllm --torch-backend=auto

# Some models may require the signing of agreements
hf auth login

# Download and serve the models
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 --port 8911
vllm serve mistralai/Ministral-8B-Instruct-2410 --port 8912
vllm serve microsoft/Phi-3-medium-128k-instruct --port 8913
```

#### Multiprocessing

All experimental scripts support multiprocessing. Setting `multiprocessing_workers` to a value greater than zero enables parallel execution. However, during the first run, model downloads may occur; therefore, it is recommended to first run with `multiprocessing_workers = 0` to verify correctness, and then restart the experiment with multiprocessing enabled. 

To avoid console clutter caused by simultaneous process outputs, the standard output is suppressed during execution. Only summary messages are displayed before and after the experiment. Some models may print progress logs during their initial download phase, which can temporarily dominate the console output. If the model download has completed but no new logs appear, the experiment is likely already in progress.

## Dataset Access

The EvoEmo dataset is provided in this repository at `data/evo_emo.json`.  

For external Python projects, data loading utilities are available in `src/lib/shared/data_provider`.

## Running the Code

After completing the environment setup, experiments can be executed by running the corresponding scripts located in the exe directory. For example, to evaluate the question-answering task using the Mistral-8B model with full dialogue history, run:

```sh
python ./src/exe/qa/qa_mistral8b_full.py
```

By default, a directory named by current datetime will be created in `./outputs/qa_mistral8b_full`. Result of each seeker will be saved as a `csv` file in a sub-directory named with the seeker's id. The contents of this file will be updated in real time to show the current progress. After running, you can check `exception.csv` to see if any exceptions occurred in each process; the final results of all seekers are merged into `result.csv`.

### Script Configuration

Besides `common_configurations.py`, to ensure that configuration can be modified independently of other scripts, there will be a `Config` class in any executable script. Its scope is limited to this script.

### Experiment List

Here is a list showing the execute scripts, and its relationship to our manuscript.

#### Table 3

<table>
    <tr>
        <th rowspan="2">Category</th>
        <th rowspan="2">Model</th>
        <th colspan="6">F1 Score (%) ↑</th>
        <th colspan="6">BERTScore (%) ↑</th>
        <th colspan="6">LLM-as-Judge (0-2) ↑</th>
    </tr>
    <tr>
        <th>IE</th>
        <th>TR</th>
        <th>CD</th>
        <th>Abs</th>
        <th>UM</th>
        <th>All</th>
        <th>IE</th>
        <th>TR</th>
        <th>CD</th>
        <th>Abs</th>
        <th>UM</th>
        <th>All</th>
        <th>IE</th>
        <th>TR</th>
        <th>CD</th>
        <th>Abs</th>
        <th>UM</th>
        <th>All</th>
    </tr>
    <tr>
        <td rowspan="3">Base</td>
        <td>Mistral-8B</td>
        <td colspan="18"><code>exe/qa/qa_mistral8b_full.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium</td>
        <td colspan="18"><code>exe/qa/qa_phi3_full.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B</td>
        <td colspan="18"><code>exe/qa/qa_mistral24b_full.py</code></td>
    </tr>
    <tr>
        <td rowspan="3">Base + RAG</td>
        <td>Mistral-8B + RAG</td>
        <td colspan="18"><code>exe/qa/qa_mistral8b_rag.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium + RAG</td>
        <td colspan="18"><code>exe/qa/qa_phi3_rag.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B + RAG</td>
        <td colspan="18"><code>exe/qa/qa_mistral24b_rag.py</code></td>
    </tr>
    <tr>
        <td rowspan="2">Commercial</td>
        <td>GPT-3.5-turbo(4K)</td>
        <td colspan="18"><code>exe/qa/qa_gpt35turbo_full.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o(16K)</td>
        <td colspan="18"><code>exe/qa/qa_gpt4o_full.py</code></td>
    </tr>
    <tr>
        <td rowspan="2">Commercial + RAG</td>
        <td>GPT-3.5-turbo + RAG</td>
        <td colspan="18"><code>exe/qa/qa_gpt35turbo_rag.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o + RAG</td>
        <td colspan="18"><code>exe/qa/qa_gpt4o_rag.py</code></td>
    </tr>
</table>

#### Table 4

<table>
    <tr>
        <th rowspan="2">Retrieval Granularity</th>
        <th rowspan="2">Top-k</th>
        <th colspan="3">Answer Prediction</th>
        <th colspan="2">Retrieval Accuracy</th>
    </tr>
    <tr>
        <th>F1 Score (%) ↑</th>
        <th>BERTScore (%) ↑</th>
        <th>LLM-as-Judge (0-2) ↑</th>
        <th>R@k (%) ↑</th>
        <th>NDCG@k (0-2) ↑</th>
    </tr>
    <tr>
        <td rowspan="3">Turn-level</td>
        <td>10</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_rag_turn_10.py</code></td>
        <td rowspan="3" colspan="2"><code>exe/qa_retrieval/qa_retrieval_turn.py</code></td>
    </tr>
    <tr>
        <td>20</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_rag_turn_20.py</code></td>
    </tr>
    <tr>
        <td>30</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_rag_turn_30.py</code></td>
    </tr>
    <tr>
        <td rowspan="3">Round-level</td>
        <td>5</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_rag_round_5.py</code></td>
        <td rowspan="3" colspan="2"><code>exe/qa_retrieval/qa_retrieval_round.py</code></td>
    </tr>
    <tr>
        <td>10</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_rag_round_10.py</code></td>
    </tr>
    <tr>
        <td>15</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_rag_round_15.py</code></td>
    </tr>
    <tr>
        <td rowspan="3">session-level</td>
        <td>2</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_rag_session_2.py</code></td>
        <td rowspan="3" colspan="2"><code>exe/qa_retrieval/qa_retrieval_session.py</code></td>
    </tr>
    <tr>
        <td>4</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_rag.py</code></td>
    </tr>
    <tr>
        <td>8</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_rag_session_8.py</code></td>
    </tr>
</table>

#### Table 5

<table>
    <tr>
        <th>Model</th>
        <th>Context</th>
        <th>F1 Score ↑</th>
        <th>BERTScore ↑</th>
        <th>LLM-as-Judge ↑</th>
    </tr>
    <tr>
        <td rowspan="4">Mistral-8B</td>
        <td>2K</td>
        <td colspan="3"><code>exe/qa/qa_mistral8b_full_2k.py</code></td>
    </tr>
    <tr>
        <td>4K</td>
        <td colspan="3"><code>exe/qa/qa_mistral8b_full_4k.py</code></td>
    </tr>
    <tr>
        <td>8K</td>
        <td colspan="3"><code>exe/qa/qa_mistral8b_full_8k.py</code></td>
    </tr>
    <tr>
        <td>20K</td>
        <td colspan="3"><code>exe/qa/qa_mistral8b_full.py</code></td>
    </tr>
    <tr>
        <td rowspan="4">Mistral-24B</td>
        <td>2K</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_full_2k.py</code></td>
    </tr>
    <tr>
        <td>4K</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_full_4k.py</code></td>
    </tr>
    <tr>
        <td>8K</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_full_8k.py</code></td>
    </tr>
    <tr>
        <td>20K</td>
        <td colspan="3"><code>exe/qa/qa_mistral24b_full.py</code></td>
    </tr>
</table>

#### Table 6

<table>
    <tr>
        <th rowspan="2">Category</th>
        <th rowspan="2">Model</th>
        <th colspan="3">ROUGE (%) ↑</th>
        <th colspan="3">Event-based Metrics (%) ↑</th>
        <th rowspan="2">LLM Score (0-5) ↑</th>
    </tr>
    <tr>
        <th>ROUGE-1</th>
        <th>ROUGE-2</th>
        <th>ROUGE-L</th>
        <th>Precision</th>
        <th>Recall</th>
        <th>F1</th>
    </tr>
    <tr>
        <td rowspan="3">Base</td>
        <td>Mistral-8B</td>
        <td colspan="7"><code>exe/sum/sum_mistral8b_full.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium</td>
        <td colspan="7"><code>exe/sum/sum_phi3_full.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B</td>
        <td colspan="7"><code>exe/sum/sum_mistral24b_full.py</code></td>
    </tr>
    <tr>
        <td rowspan="3">Base + RAG</td>
        <td>Mistral-8B + RAG</td>
        <td colspan="7"><code>exe/sum/sum_mistral8b_rag.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium + RAG</td>
        <td colspan="7"><code>exe/sum/sum_phi3_rag.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B + RAG</td>
        <td colspan="7"><code>exe/sum/sum_mistral24b_rag.py</code></td>
    </tr>
    <tr>
        <td rowspan="2">Commercial</td>
        <td>GPT-3.5-turbo(4K)</td>
        <td colspan="7"><code>exe/sum/sum_gpt35turbo_full.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o(16K)</td>
        <td colspan="7"><code>exe/sum/sum_gpt4o_full.py</code></td>
    </tr>
    <tr>
        <td rowspan="2">Commercial + RAG</td>
        <td>GPT-3.5-turbo + RAG</td>
        <td colspan="7"><code>exe/sum/sum_gpt35turbo_rag.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o + RAG</td>
        <td colspan="7"><code>exe/sum/sum_gpt4o_rag.py</code></td>
    </tr>
</table>

#### Table 7

<table>
    <tr>
        <th>Memory Setting</th>
        <th>Model</th>
        <th>Recall ↑</th>
        <th>Weighted Score ↑</th>
    </tr>
    <tr>
        <td rowspan="5">No-Mem.</td>
        <td>Mistral-8B</td>
        <td colspan="2"><code>exe/dg/dg_mistral8b.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium</td>
        <td colspan="2"><code>exe/dg/dg_phi3.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B</td>
        <td colspan="2"><code>exe/dg/dg_mistral24b.py</code></td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo</td>
        <td colspan="2"><code>exe/dg/dg_gpt35turbo.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o</td>
        <td colspan="2"><code>exe/dg/dg_gpt4o.py</code></td>
    </tr>
    <tr>
        <td rowspan="5">Full-Hist.</td>
        <td>Mistral-8B</td>
        <td colspan="2"><code>exe/dg/dg_mistral8b_full.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium</td>
        <td colspan="2"><code>exe/dg/dg_phi3_full.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B</td>
        <td colspan="2"><code>exe/dg/dg_mistral24b_full.py</code></td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo</td>
        <td colspan="2"><code>exe/dg/dg_gpt35turbo_full.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o</td>
        <td colspan="2"><code>exe/dg/dg_gpt4o_full.py</code></td>
    </tr>
    <tr>
        <td rowspan="5">RAG</td>
        <td>Mistral-8B</td>
        <td colspan="2"><code>exe/dg/dg_mistral8b_rag.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium</td>
        <td colspan="2"><code>exe/dg/dg_phi3_rag.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B</td>
        <td colspan="2"><code>exe/dg/dg_mistral24b_rag.py</code></td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo</td>
        <td colspan="2"><code>exe/dg/dg_gpt35turbo_rag.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o</td>
        <td colspan="2"><code>exe/dg/dg_gpt4o_rag.py</code></td>
    </tr>
</table>

#### Table 8

<table>
    <tr>
        <th>Memory Setting</th>
        <th>Model</th>
        <th>LT-Mem. ↑</th>
        <th>Pers. ↑</th>
        <th>ES ↑</th>
    </tr>
    <tr>
        <td rowspan="5">No-Mem.</td>
        <td>Mistral-8B</td>
        <td colspan="3"><code>exe/dg/dg_mistral8b.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium</td>
        <td colspan="3"><code>exe/dg/dg_phi3.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B</td>
        <td colspan="3"><code>exe/dg/dg_mistral24b.py</code></td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo</td>
        <td colspan="3"><code>exe/dg/dg_gpt35turbo.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o</td>
        <td colspan="3"><code>exe/dg/dg_gpt4o.py</code></td>
    </tr>
    <tr>
        <td rowspan="5">Full-Hist.</td>
        <td>Mistral-8B</td>
        <td colspan="3"><code>exe/dg/dg_mistral8b_full.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium</td>
        <td colspan="3"><code>exe/dg/dg_phi3_full.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B</td>
        <td colspan="3"><code>exe/dg/dg_mistral24b_full.py</code></td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo</td>
        <td colspan="3"><code>exe/dg/dg_gpt35turbo_full.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o</td>
        <td colspan="3"><code>exe/dg/dg_gpt4o_full.py</code></td>
    </tr>
    <tr>
        <td rowspan="5">RAG</td>
        <td>Mistral-8B</td>
        <td colspan="3"><code>exe/dg/dg_mistral8b_rag.py</code></td>
    </tr>
    <tr>
        <td>Phi-3-Medium</td>
        <td colspan="3"><code>exe/dg/dg_phi3_rag.py</code></td>
    </tr>
    <tr>
        <td>Mistral-24B</td>
        <td colspan="3"><code>exe/dg/dg_mistral24b_rag.py</code></td>
    </tr>
    <tr>
        <td>GPT-3.5-turbo</td>
        <td colspan="3"><code>exe/dg/dg_gpt35turbo_rag.py</code></td>
    </tr>
    <tr>
        <td>GPT-4o</td>
        <td colspan="3"><code>exe/dg/dg_gpt4o_rag.py</code></td>
    </tr>
</table>