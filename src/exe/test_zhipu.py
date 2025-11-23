from langchain_core.messages import HumanMessage
from exe import common_configurations
from lib.shared.basic_tools.chat_logger import ChatLogger
from lib.shared.basic_tools.output_path_builder import OutputPathBuilder
from lib.shared.chat_models.openai_model import OpenaiModel
from lib.shared.chat_models.vllm_model import VllmModel


m = VllmModel("https://open.bigmodel.cn/api/paas/v4",
              "glm-4.5-flash",
              "c5afbe32f4314fe3adc91baaa3702e60.y4JMbe2WDWDVDMeV", 
              ChatLogger(OutputPathBuilder.exe_time(common_configurations.output_directory)),
              [1, 2, 4, 16],
              8000)

print(m.generate([HumanMessage("Who are you?")]))