import json
import requests
from typing import List, Dict, Any
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
import yandexcloud
from yandexcloud import SDK

class YandexGPT(LLM):
    """YandexGPT LLM wrapper for LangChain"""
    
    api_key: str = Field(...)
    folder_id: str = Field(...)
    model_uri: str = Field(default="gpt://b1g.../yandexgpt-lite")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    
    @property
    def _llm_type(self) -> str:
        return "yandexgpt"
    
    def _call(
        self,
        prompt: str,
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
    ) -> str:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": self.temperature,
                "maxTokens": str(self.max_tokens)
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }
        
        response = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["result"]["alternatives"][0]["message"]["text"]
        else:
            raise Exception(f"YandexGPT API error: {response.text}")
