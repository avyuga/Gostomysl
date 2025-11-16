from typing import List, Dict
from langchain_core.prompts.prompt import PromptTemplate
from models.yandex_llm import YandexGPT
from config import Config

class QueryAgent:
    """Agent for query transformation and enhancement"""
    
    def __init__(self):
        self.llm = YandexGPT(
            api_key=Config.YANDEX_API_KEY,
            folder_id=Config.YANDEX_FOLDER_ID,
            model_uri=Config.YANDEX_GPT_MODEL_URI
        )
        
        self.query_enhancement_prompt = PromptTemplate(
            input_variables=["query", "language"],
            template="""
            Ты - эксперт по научному поиску. Улучши и расширь поисковый запрос.
            
            Исходный запрос: {query}
            Язык запроса: {language}
            
            Сгенерируй 3-5 улучшенных вариантов запроса для поиска научных статей на ArXiv.
            Включи синонимы, связанные термины и английские переводы.
            
            Формат ответа (JSON):
            {{
                "enhanced_queries": ["запрос1", "запрос2", ...],
                "arxiv_queries": ["query1", "query2", ...],
                "keywords": ["keyword1", "keyword2", ...]
            }}
            """
        )
    
    def process_query(self, user_query: str) -> Dict:
        """Process and enhance user query"""
        language = "русский" if any(ord(c) > 127 for c in user_query) else "английский"
        
        prompt = self.query_enhancement_prompt.format(
            query=user_query,
            language=language
        )
        
        response = self.llm(prompt)
        
        try:
            import json
            result = json.loads(response.strip('```'))
        except Exception as e:
            print('process query died', str(e), flush=True)
            # Fallback if JSON parsing fails
            result = {
                "enhanced_queries": [user_query],
                "arxiv_queries": [user_query],
                "keywords": user_query.split()
            }
        
        return result
