import pathlib as _pathlib
import csdir as _csdir
import csfile as _csfile
import torch as _torch

from lib.shared.chat_models.openai_model_indicator import OpenaiModelIndicator as _OpenaiModelIndicator
from lib.shared.chat_models.vllm_human_ai_only_model_indicator import VllmHumanAiOnlyModelIndicator as _VllmHumanAiOnlyModelIndicator
from lib.shared.chat_models.vllm_model_indicator import VllmModelIndicator as _VllmModelIndicator


# Setting to False ensures that the GPUs are not used.
prefer_cuda: bool = _torch.cuda.is_available()

# Parameters to access gpt 3.5 turbo.
gpt35turbo: _OpenaiModelIndicator = _OpenaiModelIndicator("gpt-3.5-turbo", 
                                                          _csfile.read_all_text("./secrets/open_ai_api_key.txt"))

# Parameters to access gpt 4o.
gpt4o: _OpenaiModelIndicator = _OpenaiModelIndicator("gpt-4o", 
                                                     _csfile.read_all_text("./secrets/open_ai_api_key.txt"))

# This is only used in this configuration file.
# If your LLMs are served on the same host, you can adjust all of them with this.
_vllm_host: str = "http://127.0.0.1"


# Parameters to access Mistral 24B.
mistral24b: _VllmModelIndicator = _VllmModelIndicator(f"{_vllm_host}:8911/v1", 
                                                      "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                                                      _csfile.read_all_text("./secrets/vllm_api_key.txt"),
                                                      4000)

# Parameters to access Mistral 8B.
mistral8b: _VllmHumanAiOnlyModelIndicator = _VllmHumanAiOnlyModelIndicator(f"{_vllm_host}:8912/v1", 
                                                                           "mistralai/Ministral-8B-Instruct-2410",
                                                                           _csfile.read_all_text("./secrets/vllm_api_key.txt"),
                                                                           4000)

# Parameters to access Phi 3 Medium.
phi3: _VllmModelIndicator = _VllmModelIndicator(f"{_vllm_host}:8913/v1", 
                                                "microsoft/Phi-3-medium-128k-instruct",
                                                _csfile.read_all_text("./secrets/vllm_api_key.txt"),
                                                4000)

# Experiments will output to a sub-directory of this directory.
output_directory: _pathlib.Path = _csdir.create_directory("./outputs")

# Path of our dataset.
data_path: _pathlib.Path = _pathlib.Path("./data/evo_emo.json")

# Setting to a value greater than zero will enable multiprocessing.
# Setting to zero will disable it, which can be helpful in certain situations.
# (Setting to one will still enable multiprocessing, but it generally does not offer any benefits.)
# (It is recommended to disable multiprocessing on the first run, as some models may need to be downloaded.)
multiprocessing_workers: int = 0
