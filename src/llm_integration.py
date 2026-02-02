"""
与不同LLM集成的完整示例
展示如何将Google搜索工具与各种大模型结合使用
"""

from search_google_tool import search_with_serpapi
import os
from openai import OpenAI

class LLMWithSearch:
    """带搜索功能的LLM助手"""
    
    def __init__(self):
        # self.search_tool = search_tool
        #  # 用户问题
        # question, num_results = "pubchem数据库访问方式",10
        # search_tool = search_with_serpapi(question, num_results) 
        pass
    
    def answer_with_search(self, question: str, num_results: int = 3) -> dict:
        """
        使用搜索增强回答问题
        
        Args:
            question: 用户问题
            num_results: 搜索结果数量
            
        Returns:
            包含搜索结果和提示词的字典
        """
        # 执行搜索
        search_results = search_with_serpapi(
            question, 
            num_results=num_results
        )
        
        # 构建提示词
        prompt = self._build_prompt(question, search_results)
        
        return {
            "question": question,
            "search_results": search_results,
            "prompt": prompt
        }
    
    def _build_prompt(self, question: str, search_results: str) -> str:
        """构建发送给LLM的提示词"""
        return f"""你是一个专业的AI助手。请基于以下从Google搜索获取的最新信息来回答用户的问题。
            搜索结果:
            {search_results}

            用户问题: {question}

            要求与输入返回:
            1. 若是查询到相关数据库的资源链接、下载方式或者API接口访问等获取资源的方式请生成python代码去下载，`def fetch_structure_from_mp(material_id: str, api_key: Optional[str] = None) -> dict:`
            2. 若查询到知乎等专业知识平台有人分享的使用总结请你梳理其中的内容给出准确、全面的回答
            3. 你给的输出应该包含完整的信息与可供使用的python代码

            你的回答:"""


# ===== 示例1: 与OpenAI GPT集成 =====
def request_with_openai():
    """与OpenAI GPT集成示例"""
    print("=" * 60)
    print("OpenAI GPT集成")
    print("=" * 60)
    
    try:
        # 创建助手
        assistant = LLMWithSearch()
        question = "pubchem数据库使用与下载方式"
        
        # 获取搜索增强的提示词
        result = assistant.answer_with_search(question, num_results=100)
        
        # 调用OpenAI API
       
        client = OpenAI(
            api_key="sk-dkqEVEHBBbWtdmwLeyc0xyGxfcNTTHTESX5cmr4jxIh6S00M",
            base_url="https://zjuapi.com/v1"
        )
        response = client.chat.completions.create(
            model="claude-sonnet-4-5-20250929",
            messages=[
                {"role": "system", "content": "你是一个专业的AI研究助手"},
                {"role": "user", "content": result["prompt"]}
            ],
            temperature=0.7
        )
        
        print(f"\n问题: {question}")
        print(f"\n搜索到的信息:\n{result['search_results']}")
        print(f"\nGPT回答:\n{response.choices[0].message.content}")
        
    except ImportError:
        print("需要安装openai库: pip install openai")
    except Exception as e:
        print(f"错误: {e}")



# ===== 示例2: 批量问题处理 =====
def batch_questions_processing():
    """批量处理多个问题"""
    print("\n" + "=" * 60)
    print("示例5: 批量问题处理")
    print("=" * 60)
    
    # 初始化搜索工具
    search_tool = GoogleSearchTool(
        api_key=os.getenv("GOOGLE_API_KEY", "your_google_api_key"),
        search_engine_id=os.getenv("SEARCH_ENGINE_ID", "your_engine_id")
    )
    
    assistant = LLMWithSearch(search_tool)
    
    # 批量问题
    questions = [
        "什么是大语言模型?",
        "Python和JavaScript的主要区别",
        "如何学习机器学习?"
    ]
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n处理问题 {i}/{len(questions)}: {question}")
        result = assistant.answer_with_search(question, num_results=2)
        results.append(result)
        print(f"✓ 完成")
    
    # 保存结果
    print("\n所有问题处理完成!")
    print(f"共收集了 {len(results)} 组搜索结果")
    
    return results


def main():
    """主函数 - 运行所有示例"""
    
    print("\n" + "=" * 60)
    print("Google搜索工具 - LLM集成示例")
    print("=" * 60)
    
    print("\n注意: 运行前请设置环境变量:")
    print("  - GOOGLE_API_KEY: Google API密钥")
    print("  - SEARCH_ENGINE_ID: 搜索引擎ID")
    print("  - OPENAI_API_KEY: OpenAI API密钥(可选)")
    print("  - ANTHROPIC_API_KEY: Anthropic API密钥(可选)")
    
    # 取消注释你想运行的示例
    
    request_with_openai()
    # example_with_claude()
    # example_with_local_model()
    # interactive_assistant()
    # batch_questions_processing()
    
    print("\n提示: 取消注释main()中的函数调用来运行相应示例")


if __name__ == "__main__":
    main()