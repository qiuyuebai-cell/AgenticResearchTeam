import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from knowledge_base_tool import knowledge_base_search
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
google_key = os.getenv('GOOGLE_API_KEY')
zhipu_key = os.getenv('ZHIPUAI_API_KEY')
# 初始化模型
llm = 'gemini/gemini-2.5-pro'
# llm = 'zhipu/glm-4'

# 1.检索专家
researcher = Agent(
    role='高级知识检索工程师',
    goal='根据用户的问题,在知识库中定位并提取所有相关的、未经加工的原始信息片段。',
    backstory='你是一个精通信息检索的专家',
    verbose=True,
    allow_delegation=False,
    tools=[knowledge_base_search],
    llm=llm
)
# 2.内容分析师
analyst = Agent(
    role='《三体》宇宙首席分析师',
    goal='基于检索专家提供的背景资料，为用户的原始问题撰写一份全面、深刻且完全忠于原文的分析报告。',
    backstory="""你是一位世界级的《三体》学者...""",  # 省略部分背景描述
    verbose=True,  # 显示详细输出
    allow_delegation=False,  # 不能将任务委托给其他代理
    llm=llm
)

# 任务定义
research_task = Task(
    description='对用户提出的问题 "{topic}" 进行深入研究，并从知识库中找到所有相关信息。',
    expected_output='一份包含所有从知识库中检索到的原始文本片段的文档...',  # 省略
    agent=researcher
)

analysis_task = Task(
    description="""请基于以下背景资料，深入分析并回答用户的原始问题：'{topic}'。
      你的回答需要结构清晰，全面深刻，并且完全基于所提供的资料。
      背景资料:
      {{context}}
      """,  # 省略
    expected_output='一份格式精良、逻辑清晰、完全基于所提供资料的最终回答。',
    agent=analyst,
    context=[research_task]
)


# 创建Crew
def create_crew_ai(topic: str):
    """根据给定的主题创建AI Crew实例"""
    research_task.description = f'对用户提出的问题 "{topic}" 进行深入研究，并从知识库中找到所有相关信息。'
    analysis_task.description = f"""请基于以下背景资料，深入分析并回答用户的原始问题：'{topic}'。
      你的回答需要结构清晰，全面深刻，并且完全基于所提供的资料。
      背景资料:
      {{context}}'
        """

    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=True
    )

    return crew


if __name__ == '__main__':
    user_topic = input('你好！我是Agentic Research Team, 有什么可以帮助你的吗？')
    if user_topic:

        ai_crew = create_crew_ai(user_topic)


        # 定义一个带重试逻辑的函数来启动crew
        @retry(
            stop=stop_after_attempt(6),  # 最多重试6次
            wait=wait_exponential(multiplier=1, min=10, max=30)  # 等待时间：2s, 4s, 8s...
        )
        def retry_crew():
            results = ai_crew.kickoff()
            return results


        try:
            final_results = retry_crew()
            print('\n\n###################################')
            print('本地知识库检索分析结果如下')
            print('###################################\n')
            print(final_results)

        except Exception as e:
            print('error!')
