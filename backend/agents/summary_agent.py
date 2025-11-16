from typing import List, Dict
from models.yandex_llm import YandexGPT
from config import Config
import asyncio

class SummaryAgent:
    """Agent for summarizing papers"""
    
    def __init__(self):
        self.llm = YandexGPT(
            api_key=Config.YANDEX_API_KEY,
            folder_id=Config.YANDEX_FOLDER_ID,
            max_tokens=1000,
            model_uri=Config.YANDEX_GPT_MODEL_URI
        )
        
        self.summary_prompt = """
        Создай краткую суммаризацию научной статьи на русском языке.
        
        Название: {title}
        Авторы: {authors}
        Аннотация: {abstract}
        
        Суммаризация должна включать:
        1. Основную проблему/задачу
        2. Предложенный метод/подход
        3. Основные результаты
        4. Практическая значимость
        
        Ответ на русском языке, максимум 200 слов.
        """
    
    async def summarize_paper(self, paper: Dict) -> Dict:
        """Summarize a single paper"""
        prompt = self.summary_prompt.format(
            title=paper['title'],
            authors=', '.join(paper['authors'][:3]),
            abstract=paper['summary'][:1500]
        )
        
        summary = self.llm(prompt)
        
        return {
            **paper,
            'ru_summary': summary
        }
    
    async def summarize_papers(self, papers: List[Dict]) -> List[Dict]:
        """Summarize multiple papers"""
        tasks = [self.summarize_paper(paper) for paper in papers]
        summarized = await asyncio.gather(*tasks)
        return summarized
    
    async def filter_relevant(self, papers: List[Dict], query: str) -> List[Dict]:
        """Filter out irrelevant papers based on summaries"""
        filter_prompt = """
        Определи, релевантна ли статья для запроса.
        
        Запрос: {query}
        
        Суммаризация статьи:
        {summary}
        
        Ответь: ДА или НЕТ
        """
        
        relevant_papers = []
        for paper in papers:
            prompt = filter_prompt.format(
                query=query,
                summary=paper.get('ru_summary', paper['summary'][:500])
            )
            
            response = self.llm(prompt)
            
            if 'ДА' in response.upper():
                relevant_papers.append(paper)
        
        return relevant_papers
