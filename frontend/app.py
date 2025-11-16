import streamlit as st
import asyncio
import websockets
import json
from datetime import datetime

st.set_page_config(
    page_title="ArXiv Research System",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stage-complete {
        color: #4CAF50;
        font-weight: bold;
    }
    .stage-pending {
        color: #FFA500;
    }
    .stage-active {
        color: #2196F3;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 ArXiv Research System")
st.markdown("### Многоагентная система поиска и анализа научных статей")

# Sidebar
with st.sidebar:
    st.header("Настройки")
    api_url = st.text_input("Backend URL", value="ws://localhost:8000/ws/research")
    
    st.markdown("---")
    st.markdown("### О системе")
    st.markdown("""
    Эта система использует:
    - 🤖 YandexGPT для анализа
    - 📚 ArXiv для поиска статей
    - 🎯 Многоуровневое ранжирование
    - 📝 Форматирование по ГОСТ
    """)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_area(
        "Введите запрос для поиска",
        placeholder="Например: машинное обучение для обработки естественного языка",
        height=100
    )
    
    if st.button("🔍 Начать исследование", type="primary"):
        if query:
            # Progress tracking
            progress_container = st.container()
            
            # Stage indicators
            stages = {
                "query_processing": {"name": "Обработка запроса", "status": "pending"},
                "searching": {"name": "Поиск в ArXiv", "status": "pending"},
                "ranking": {"name": "Ранжирование", "status": "pending"},
                "summarizing": {"name": "Суммаризация", "status": "pending"},
                "filtering": {"name": "Фильтрация", "status": "pending"},
                "analysis": {"name": "Анализ области", "status": "pending"},
                "formatting": {"name": "Форматирование", "status": "pending"}
            }
            
            # Create placeholders for each stage
            stage_placeholders = {}
            for stage_id, stage_info in stages.items():
                stage_placeholders[stage_id] = st.empty()
            
            # Results placeholders
            results_placeholder = st.empty()
            
            async def run_research():
                try:
                    async with websockets.connect(api_url) as websocket:
                        # Send query
                        await websocket.send(json.dumps({"query": query}))
                        
                        # Receive updates
                        while True:
                            message = await websocket.recv()
                            data = json.loads(message)
                            
                            stage = data.get("stage")
                            status = data.get("status")
                            
                            if stage in stages:
                                # Update stage status
                                if status == "Complete":
                                    stages[stage]["status"] = "complete"
                                    stage_placeholders[stage].success(
                                        f"✅ {stages[stage]['name']}: Завершено"
                                    )
                                    
                                    # Show stage data
                                    if "data" in data:
                                        with st.expander(f"Результаты: {stages[stage]['name']}"):
                                            if stage == "query_processing":
                                                st.json(data["data"])
                                            elif stage == "searching":
                                                st.metric("Найдено статей", data["data"]["count"])
                                            elif stage == "ranking":
                                                for paper in data["data"]["top_papers"][:3]:
                                                    st.write(f"📄 {paper['title']}")
                                            elif stage == "summarizing":
                                                for item in data["data"]["summaries"]:
                                                    st.write(f"**{item['title']}**")
                                                    st.write(item['summary'])
                                            elif stage == "filtering":
                                                st.metric("Релевантных статей", data["data"]["relevant_count"])
                                            elif stage == "analysis":
                                                st.json(data["data"]["plan"])
                                
                                else:
                                    stages[stage]["status"] = "active"
                                    stage_placeholders[stage].info(
                                        f"⏳ {stages[stage]['name']}: {status}"
                                    )
                            
                            elif stage == "complete":
                                # Show final results
                                results_placeholder.success("🎉 Исследование завершено!")
                                
                                # Display document
                                st.markdown("---")
                                st.markdown("## 📄 Результат анализа")
                                
                                document = data["data"]["document"]
                                
                                # Create download button
                                st.download_button(
                                    label="📥 Скачать документ",
                                    data=document,
                                    file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
                                
                                # Display document
                                st.markdown(document)
                                
                                break
                            
                            elif stage == "error":
                                st.error(f"Ошибка: {status}")
                                break
                
                except Exception as e:
                    st.error(f"Ошибка подключения: {str(e)}")
            
            # Run async function
            asyncio.run(run_research())
        else:
            st.warning("Пожалуйста, введите запрос")

with col2:
    st.markdown("### 📊 Статистика")
    
    # Placeholder for statistics
    stats_container = st.container()
    with stats_container:
        st.metric("Всего обработано", "0")
        st.metric("Время выполнения", "0 сек")
        st.metric("Использовано токенов", "0")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Разработано с использованием LangGraph и YandexGPT
    </div>
    """,
    unsafe_allow_html=True
)
