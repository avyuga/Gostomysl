from typing import List, Dict
from models.yandex_llm import YandexGPT
from config import Config
import json

class AnalysisAgent:
    """Agent for creating domain analysis"""
    
    def __init__(self):
        self.llm = YandexGPT(
            api_key=Config.YANDEX_API_KEY,
            folder_id=Config.YANDEX_FOLDER_ID,
            max_tokens=3000
        )
    
    async def create_analysis_plan(self, papers: List[Dict], query: str) -> Dict:
        """Create analysis plan based on papers"""
        
        papers_info = "\n".join([
            f"{i+1}. {p['title']} - {p.get('ru_summary', p['summary'][:200])}"
            for i, p in enumerate(papers[:10])
        ])
        
        plan_prompt = f"""
        Создай план анализа предметной области на основе статей.
        
        Запрос пользователя: {query}
        
        Найденные статьи:
        {papers_info}
        
        Создай структурированный план анализа в формате JSON:
        {{
            "title": "Название анализа",
            "sections": [
                {{
                    "title": "Название раздела",
                    "content_plan": "Что будет в разделе",
                    "papers_refs": [1, 2, 3]  // номера статей
                }}
            ],
            "conclusion_plan": "План заключения"
        }}
        """
        
        response = self.llm(plan_prompt)
        
        try:
            plan = json.loads(response)
        except:
            # Fallback plan
            plan = {
                "title": f"Анализ предметной области: {query}",
                "sections": [
                    {
                        "title": "Введение",
                        "content_plan": "Общий обзор темы",
                        "papers_refs": list(range(1, min(4, len(papers)+1)))
                    },
                    {
                        "title": "Основные подходы",
                        "content_plan": "Описание методов и подходов",
                        "papers_refs": list(range(1, min(7, len(papers)+1)))
                    },
                    {
                        "title": "Результаты и применение",
                        "content_plan": "Практические результаты",
                        "papers_refs": list(range(1, min(6, len(papers)+1)))
                    }
                ],
                "conclusion_plan": "Выводы и перспективы"
            }
        
        return plan
    
    async def write_analysis(self, plan: Dict, papers: List[Dict]) -> str:
        """Write full analysis based on plan"""
        
        analysis_parts = []
        
        # Title
        analysis_parts.append(f"# {plan['title']}\n\n")
        
        # Sections
        for section in plan['sections']:
            section_papers = [papers[i-1] for i in section['papers_refs'] if i <= len(papers)]
            
            papers_context = "\n".join([
                f"- {p['title']}: {p.get('ru_summary', p['summary'][:300])}"
                for p in section_papers
            ])
            
            section_prompt = f"""
            Напиши раздел научного анализа на русском языке.
            
            Название раздела: {section['title']}
            План раздела: {section['content_plan']}
            
            Используй информацию из статей:
            {papers_context}
            
            Напиши связный текст на 200-300 слов.
            """
            
            section_text = self.llm(section_prompt)
            analysis_parts.append(f"## {section['title']}\n\n{section_text}\n\n")
        
        # Conclusion
        conclusion_prompt = f"""
        Напиши заключение для анализа предметной области.
        
        План заключения: {plan['conclusion_plan']}
        
        Кратко суммируй основные выводы на основе анализа.
        Текст на 100-150 слов.
        """
        
        conclusion = self.llm(conclusion_prompt)
        analysis_parts.append(f"## Заключение\n\n{conclusion}\n\n")
        
        return "".join(analysis_parts)
