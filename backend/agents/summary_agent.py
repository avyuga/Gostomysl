import asyncio
import logging
import traceback
import uuid
from typing import Dict, List

import aiofiles
import aiohttp
import fitz  # PyMuPDF
from aiopath import AsyncPath
from config import Config
from models.yandex_llm import YandexGPT

logger = logging.getLogger("SummaryAgent")
logger.setLevel(logging.DEBUG)


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
        Текст статьи: {text}
        
        Суммаризация должна включать:
        1. Основную проблему/задачу
        2. Предложенный метод/подход
        3. Основные результаты
        4. Практическая значимость
        
        Ответ на русском языке, максимум 200 слов.
        """

    @staticmethod
    async def extract_full_text(paper_url: str) -> str:
        try:
            # Convert abstract URL to PDF URL
            if '/abs/' in paper_url:
                pdf_url = paper_url.replace('/abs/', '/pdf/') + '.pdf'
            else:
                pdf_url = paper_url
            
            # Download PDF to temporary file
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as resp:
                    resp.raise_for_status()
                    content = await resp.read()
            
            # Create a temporary file and write the downloaded content
            temp_file_path = AsyncPath('.').joinpath(f'{uuid.uuid4()}.pdf')
            async with aiofiles.open(temp_file_path, mode="wb") as f:
                await f.write(content)
            
            try:
                # Run synchronous extraction in an executor thread
                loop = asyncio.get_running_loop()
                doc = await loop.run_in_executor(None, fitz.open, temp_file_path)
                structured_text = await loop.run_in_executor(None, SummaryAgent._extract_structured_text, doc)
                doc.close()
            
            finally:
                # Remove temporary file
                await temp_file_path.unlink(missing_ok=True)


            if structured_text.strip():
                return structured_text
            else:
                logger.error("Could not extract text from PDF")
                return None
                    
        except aiohttp.ClientResponseError as e:
            logger.error(f"Error loading PDF: {e}")
            logger.error(traceback.format_exc())
            return None
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def _extract_structured_text(doc) -> str:
        """Extract and clean text using PyMuPDF's structure analysis with NLTK sentence tokenization."""
        all_text = ""
        # First, extract all text from all pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            all_text += page_text + "\n"
        
        return all_text
    

    async def summarize_paper(self, paper: Dict) -> Dict:
        """Summarize a single paper"""
        
        full_text = await SummaryAgent.extract_full_text(paper['id'])

        prompt = self.summary_prompt.format(
            title=paper['title'],
            authors=', '.join(paper['authors'][:3]),
            text=full_text
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

