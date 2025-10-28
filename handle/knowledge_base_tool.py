import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from crewai.tools import BaseTool

load_dotenv()
em_token = os.getenv('EM_TOKEN')
db_path = './trisolaris_db'

embeddings = HuggingFaceEndpointEmbeddings(huggingfacehub_api_token=em_token,
                                           model='sentence-transformers/all-MiniLM-L6-v2')
db = Chroma(persist_directory=db_path, embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={'k': 3})


class KnowledgeBaseTool(BaseTool):
    name: str = '本地知识库检索工具'
    description: str = """当需要在查询关于《三体》等特定知识时，使用此工具
    """

    def _run(self, query: str) -> str:
        """从知识库中搜索与查询相关的信息并返回结果
        Args:
            query (str): 要搜索的查询字符串

        Returns:
            str: 从知识库检索到的相关信息，格式化为易读的文本
        """
        relevant_docs = retriever.invoke(query)
        formatted_results = '\n\n'.join(
            [f'[知识片段{i + 1}]:\n{doc.page_content}' for i, doc in enumerate(relevant_docs)]
        )
        return f'从知识库检索到以下信息:\n{formatted_results}'


knowledge_base_search = KnowledgeBaseTool()

if __name__ == '__main__':
    test_query = '叶文洁的职位'
    results = knowledge_base_search.run(test_query)
    print(results)
