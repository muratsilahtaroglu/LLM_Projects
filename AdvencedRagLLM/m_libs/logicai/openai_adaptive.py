import json
import os
from typing import Any, Dict, List, Literal
from uuid import uuid4

from colorama import Fore, Style
import openai

from logicai._common import AlgoException
from logicai.entities import ChatResponseFunction
from logicai.logging import ChatLogger


__all__ = ["Chat"]

class Chat:
    def __init__(self, model: str = "gpt-4", history: List[Dict]|None = None, api_key: str = None, logger: ChatLogger = None, chat_name:str =None) -> None:
        self.chat_id = str(uuid4())
        self.chat_name = chat_name
        self._logger = logger or ChatLogger(None)
        self.history = history if history else []
        self._log_kwargs = {"chat_id": self.chat_id, "chat_name": self.chat_name}
        self._logger.debug(topic="init_history", **self._log_kwargs, model=model, history=json.dumps(self.history))
        self.model = model
        self.openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key

    def ask(self, question: str) -> str:
        self.add_message("user", question)
        try:
            answer = self.run()
        except Exception as e:
            answer = str(e)
        return answer

    def run(self, output_json: bool = False, tools: List[Dict] = None, **kwargs) -> ChatResponseFunction | str:
        for i in range(3):
            try:
                response = self._run(output_json=output_json, tools=tools, **kwargs)
                return response
            except json.decoder.JSONDecodeError as e:
                error_message = "ERROR: Veri JSON formatında değil!\n" + str(e)
                self._logger.error(topic="Chat.run", **self._log_kwargs, error=str(e))
                self.add_message("system", error_message)
            except Exception as e:
                error_message = "ERROR: \n" + str(e)
                self._logger.error(topic="Chat.run", **self._log_kwargs, error=str(e))
                self.add_message("system", error_message)
        error_message = "ERROR: Maalesef, yapmayı denedim ama başarısız oldum!"
        self._logger.error(topic="Chat.run", **self._log_kwargs, error=error_message)
        raise AlgoException(error_message)

    
    def _run(self, output_json: bool = False, tools: List[Dict] = None, **kwargs) -> ChatResponseFunction | str:
        api_kwargs = {
            "model": self.model,
            "messages": self.history
        }

        if tools:
            api_kwargs["functions"] = tools
            # İsterseniz burada function_call parametresini de özelleştirebilirsiniz.
            # Örneğin, belirli bir fonksiyonu çağırmak isteyebilirsiniz.
            # api_kwargs["function_call"] = {"name": "function_name"}  # Belirli bir fonksiyon adıyla çağırma
            api_kwargs["function_call"] = "auto"  # Model otomatik olarak uygun fonksiyonu seçecek

        # OpenAI API çağrısı
        response = openai.chat.completions.create(
            **api_kwargs,
            **kwargs
        )

        if output_json:
            json_response_str = response.choices[0].message.content
            self.add_message("assistant", json_response_str)
            json_response = json.loads(json_response_str)
            return json_response
        
        if hasattr(response.choices[0].message, "function_call") and response.choices[0].message.function_call:
            function_call = response.choices[0].message.function_call
            arguments = json.loads(function_call.arguments)
            name = function_call.name
            self.add_function_call({"name": name, "arguments": function_call.arguments})
            return ChatResponseFunction(name=name, arguments=arguments)
        
        answer = response.choices[0].message.content
        self.add_message("assistant", answer)
        return answer
    
    def add_message(self, role: Literal["user", "assistant", "system"], content: str, index:int=None) -> None:
        self._logger.info(topic="add_message", **self._log_kwargs, role=role, content=content, index=index)
        index = index or len(self.history)
        self.history.insert(index, {"role": role, "content": content})
    
    def add_function_call(self, function_call: dict) -> None:
        self._logger.info(topic="add_function_call", **self._log_kwargs, role="assistant", function_call=function_call)
        self.history.append({"role": "assistant", "content": None, "function_call": function_call})
    
    def add_function_result(self, function_name:str, function_result: Any) -> None:
        print(Fore.MAGENTA + f"FUNCTION RESULT: \n{function_name}:\n{function_result}\n" + Style.RESET_ALL)
        self._logger.info(topic="add_function_result", **self._log_kwargs, role="function", name= function_name, function_call=function_result)
        self.history.append({"role": "function", "name": function_name, "content": function_result})
