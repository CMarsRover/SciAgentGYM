from openai import OpenAI
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
import base64
import mimetypes
import os
import re
import argparse
from py_coding_extract import extract_number_from_path, extract_code_with_filename, save_extracted_code
from pydantic import BaseModel, Field 

# 导入自定义的搜索工具
from llm_integration import LLMWithSearch
SEARCH_AVAILABLE = True


class FirstRoundAnalysis(BaseModel):
    """第一轮分析结果的数据结构"""
    # 图片相关字段
    image_description: Optional[str] = Field(None, description="图片内容、细节和关键信息的详细描述")
    identified_concepts: Optional[List[str]] = Field(None, description="从图片中识别出的核心概念列表")
    
    # 问题分析字段（兼容旧格式）
    image_analysis: Optional[str] = Field(None, description="图片内容、细节和关键信息的详细描述（兼容旧格式）")
    problem_analysis: Optional[str] = Field(None, description="问题分析内容")
    
    # 搜索相关字段
    need_search: bool = Field(..., description="是否需要网络搜索获取额外信息")
    search_query: Optional[str] = Field(None, description="如果需要搜索，提供具体的搜索关键词和目的，组织成一句话")
    search_reason: Optional[str] = Field(None, description="需要搜索的原因，比如专业数据库信息、学科专属python包或专家经验")
    
    def get_analysis_content(self) -> str:
        """获取分析内容，优先返回image_description，然后是image_analysis，最后是problem_analysis"""
        return (self.image_description or 
                self.image_analysis or 
                self.problem_analysis or 
                "无分析内容")
    
    def get_concepts(self) -> List[str]:
        """获取识别的概念列表"""
        return self.identified_concepts or []

@dataclass
class ConversationTurn:
    """对话轮次数据结构"""
    role: str  # "user", "assistant", 或 "system"
    content: str
    images: List[str] = None  # 图片路径列表，默认为空列表
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化后处理，确保images是列表"""
        if self.images is None:
            self.images = []

def create_sci_tool_conversation(query: str, image_paths: List[str] = None, search_results: str = None, first_round_result: str = None) -> List[ConversationTurn]:
    """创建科学工具生成的两轮对话"""
    
    # 确保image_paths是列表格式
    if image_paths is None:
        image_paths = []
    elif isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # 第一轮：图片识别和描述
    first_round_system = "你是一个学科专家能识别各种学科专业的图片，请对图片的内容、细节与关键信息作出描述。"
    first_round_user = f"请分析以下图片并描述其内容、细节和关键信息：\n问题：{query}"
    
    # 读取第二轮的system prompt
    with open("prompts/SystemPrompt_Science_Toolkit.md", 'r', encoding='utf-8') as f:
        second_round_system = f.read()
    
    # 构建第二轮的用户消息
    second_round_user_parts = [
        f"科学问题：{query}",
        f"图片分析结果：{first_round_result if first_round_result else '[等待第一轮分析结果]'}",
        f"搜索工具结果：{search_results if search_results else '[等待搜索工具结果]'}"
    ]
    second_round_user = "\n\n".join(second_round_user_parts)
    
    return [
        # 第一轮对话
        ConversationTurn("system", first_round_system),
        ConversationTurn("user", first_round_user, images=image_paths),
        
        # 第二轮对话
        ConversationTurn("system", second_round_system),
        ConversationTurn("user", second_round_user, images=image_paths)
    ]

def convert_turns_to_api_messages(conversation_history: List[ConversationTurn]) -> List[Dict[str, Any]]:
    """将ConversationTurn列表转换为API所需的messages格式，支持图片"""
    api_messages = []
    for turn in conversation_history:
        message_content = []
        
        # 添加文本内容
        if turn.content:
            message_content.append({
                "type": "text",
                "text": turn.content
            })
        
        # 添加图片内容
        if turn.images:
            # 确保turn.images是列表格式
            images_list = turn.images
            if isinstance(images_list, str):
                images_list = [images_list]
            elif images_list is None:
                images_list = []
            
            for image_path in images_list:
                if image_path:  # 确保图片路径不为空
                    try:
                        # 编码图片为base64
                        import base64
                        with open(image_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        # 获取图片MIME类型
                        import mimetypes
                        mime_type, _ = mimetypes.guess_type(image_path)
                        if not mime_type:
                            mime_type = "image/jpeg"  # 默认类型
                        
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        })
                    except Exception as e:
                        print(f"图片处理失败 {image_path}: {e}")
        
        # 构建消息
        if message_content:
            api_messages.append({
                "role": turn.role,
                "content": message_content
            })
        elif turn.role == "system":
            # system 消息不需要图片，直接使用文本内容
            api_messages.append({
                "role": "system",
                "content": turn.content
            })
 
    return api_messages

def multi_turn_chat(conversation_turns: List[ConversationTurn], stream: bool = False) -> List[ConversationTurn]:
    """多轮对话 - 遍历传入的ConversationTurn列表
    
    Args:
        conversation_turns: 对话轮次列表
        stream: 是否使用流式输出（默认False，用于长响应避免超时）
    """
    conversation_history = []  # 存储完整的对话历史
    # 创建客户端
    client = OpenAI(
            api_key="sk-bhcvvaKeyQyguQ0dMUxjXUHJ3LtZPvASsLx9YnXtLdNhwD0R",
            base_url="https://api.boyuerichdata.opensphereai.com/v1"
        )
    
    
    first_round_result = None  # 存储第一轮的结果
    
    for turn in conversation_turns:
        # 添加当前轮次到对话历史
        conversation_history.append(turn)
        
        # 如果是用户消息，需要获取助手回复
        if turn.role == "user":
            try:
                # 将ConversationTurn对象转换为API所需的格式
                api_messages = convert_turns_to_api_messages(conversation_history)
                
                if stream:
                    # 流式输出模式
                    print("🔄 使用流式输出模式（避免超时）...")
                    assistant_reply = _stream_chat_completion(client, api_messages)
                else:
                    # 普通模式
                    response = client.chat.completions.create(
                        model="anthropic/claude-sonnet-4.5",
                        messages=api_messages,
                        temperature=0.2,
                        max_tokens=64000
                    ) 
                    assistant_reply = response.choices[0].message.content
                    print(f"Assistant: {assistant_reply[:200]}...")
                
                # 创建助手对话轮次并添加到历史
                assistant_turn = ConversationTurn(
                    role="assistant",
                    content=assistant_reply,
                    images=[],  # 助手回复暂时不支持图片
                    metadata={"timestamp": None}
                )
                conversation_history.append(assistant_turn)
                
                # 如果是第一轮，保存结果用于第二轮
                if first_round_result is None:
                    first_round_result = assistant_reply
                    print(f"第一轮结果已保存: {first_round_result[:100]}...")
                
            except Exception as e:
                print(f"请求失败: {e}")
                # 可以选择继续处理下一个turn或者break
                break
    
    return conversation_history

def _stream_chat_completion(client, api_messages, model: str = "claude-sonnet-4-20250514", temperature: float = 0.2, max_tokens: int = 64000) -> str:
    """流式输出处理函数
    
    Args:
        client: OpenAI客户端
        api_messages: API消息列表
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大token数
        
    Returns:
        完整的助手回复内容
    """
    print("\n" + "="*60)
    print("开始流式接收响应...")
    print("="*60)
    
    full_content = ""
    chunk_count = 0
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True  # 启用流式输出
        )
        
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content = delta.content
                    full_content += content
                    chunk_count += 1
                    
                    # 实时打印内容（每10个chunk打印一次进度，避免刷屏）
                    if chunk_count % 10 == 0:
                        print(f"📥 已接收 {chunk_count} 个数据块，当前长度: {len(full_content)} 字符", end='\r')
                    
                    # 实时输出内容（可选：如果希望看到实时内容，取消下面的注释）
                    # print(content, end='', flush=True)
        
        print(f"\n✅ 流式接收完成！共接收 {chunk_count} 个数据块，总长度: {len(full_content)} 字符")
        print("="*60 + "\n")
        
        return full_content
        
    except Exception as e:
        print(f"\n❌ 流式输出过程中出错: {e}")
        # 如果流式输出失败，返回已接收的内容
        if full_content:
            print(f"⚠ 返回已接收的部分内容（{len(full_content)} 字符）")
            return full_content
        else:
            raise

def display_conversation_history(conversation_history: List[ConversationTurn]):
    """显示对话历史（用于调试）"""
    print("\n=== 对话历史 ===")
    for i, turn in enumerate(conversation_history):
        print(f"{i+1}. {turn.role}: {turn.content[:100]}{'...' if len(turn.content) > 100 else ''}")
        if turn.images:
            print(f"   图片: {turn.images}")
        if turn.metadata:
            print(f"   元数据: {turn.metadata}")
    print("================\n") 
 

def parse_first_round_result(first_round_result: str) -> FirstRoundAnalysis:
    """解析第一轮结果，使用Pydantic验证JSON格式"""
    if not first_round_result:
        return FirstRoundAnalysis(
            image_analysis="无分析结果",
            need_search=False,
            search_query=None,
            search_reason=None
        )
    
    try:
        # 尝试从结果中提取JSON
        import re
        
        # 查找JSON部分 - 支持多种格式
        json_match = re.search(r'\{[^{}]*"(?:image_description|image_analysis|problem_analysis)"[^{}]*\}', first_round_result, re.DOTALL)
        if not json_match:
            # 尝试更宽松的匹配
            json_match = re.search(r'\{.*\}', first_round_result, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            print(f"提取的JSON: {json_str[:200]}...")
            
            # 解析JSON
            data = json.loads(json_str) 
            print("!debug!")
            print(data)
        
            
            # 使用Pydantic验证
            analysis = FirstRoundAnalysis(**data)
            print(f"解析成功: need_search={analysis.need_search}, search_query={analysis.search_query}")
           
            return analysis
        else:
            # 如果没有找到JSON，使用传统方法解析
            print("警告：未找到JSON格式，使用传统解析方法")
            return _parse_legacy_format(first_round_result)
            
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print("尝试使用传统解析方法")
        return _parse_legacy_format(first_round_result)
    except Exception as e:
        print(f"解析第一轮结果时出错: {e}")
        return _parse_legacy_format(first_round_result)

def _parse_legacy_format(first_round_result: str) -> FirstRoundAnalysis:
    """传统格式解析（备用方法）"""
    # 简单的关键词检测
    search_indicators = ["需要搜索", "需要网络搜索", "需要查询", "需要获取", "搜索关键词", "搜索目的"]
    need_search = any(indicator in first_round_result for indicator in search_indicators)
    
    # 提取搜索关键词
    search_query = ""
    if need_search:
        lines = first_round_result.split('\n')
        for line in lines:
            if "搜索关键词" in line or "搜索目的" in line:
                search_query = line.strip()
                break
    
    return FirstRoundAnalysis(
        image_analysis=first_round_result[:200] + "..." if len(first_round_result) > 200 else first_round_result,
        need_search=need_search,
        search_query=search_query if search_query else None,
        search_reason="传统解析方法" if need_search else None
    )

def perform_search_if_needed(query: str, first_round_result: str) -> str:
    """如果需要搜索，执行搜索"""
    # 处理 first_round_result 为 None 的情况
    if not first_round_result:
        print("第一轮结果为空，跳过搜索")
        return ""
    
    # 使用新的解析方法
    analysis = parse_first_round_result(first_round_result)
    
    if not analysis.need_search or not SEARCH_AVAILABLE:
        print("不需要搜索或搜索工具不可用")
        return ""
    
    print(f"\n=== 执行搜索 ===")
    print(f"搜索原因: {analysis.search_reason}")
    print(f"搜索查询: {analysis.search_query}")
    
    try:
        # 创建搜索助手
        search_assistant = LLMWithSearch()
        
        # 使用解析出的搜索查询
        search_query = analysis.search_query or f"{query} {analysis.search_reason}"
        
        # 执行搜索
        search_result = search_assistant.answer_with_search(search_query, num_results=5)
        
        print(f"搜索完成，获得 {len(search_result.get('search_results', ''))} 字符的搜索结果")
        print(f"搜索完成，内容是{search_result.get('search_results', '')} ")
        return search_result
        
    except Exception as e:
        print(f"搜索过程中出错: {e}")
        return ""

def generate_sci_tool(query: str, answer:str,image_paths: List[str] = None, search_results: str = None,subfield:str = None, use_stream: bool = False):
    """生成科学工具的主函数
    
    Args:
        query: 科学问题
        answer: 标准答案
        image_paths: 图片路径列表
        search_results: 预定义的搜索结果（可选）
        subfield: 学科子领域
        use_stream: 是否使用流式输出（默认False，使用非流式。设置为True可避免长响应超时）
    """
    print(f"开始生成科学工具...")
    print(f"问题: {query}")
    print(f"输出模式: {'流式' if use_stream else '非流式'}")
    # print(f"图片: {image_paths if image_paths else '无图片'}")
    
    # 第一轮：问题分析（带重试机制）
    if image_paths:
        round_name = "图片识别和描述"
    else:
        round_name = "问题分析和搜索判断"
    print(f"\n=== 第一轮：{round_name} ===")
    first_round_result = None
    analysis = None
    max_retries = 3
    
    for attempt in range(max_retries):
        print(f"尝试第 {attempt + 1} 次...")
        first_round_conversations = create_first_round_conversation(query, image_paths,subfield)
        first_round_result = multi_turn_chat(first_round_conversations, stream=use_stream)
        
        # 提取第一轮的结果
        first_round_analysis = None
        for turn in first_round_result:
            if turn.role == "assistant":
                first_round_analysis = turn.content
                break
        
        print(f"\n第一轮分析结果: {first_round_analysis[:200] if first_round_analysis else 'None'}...")
        
        # 解析第一轮结果
        analysis = parse_first_round_result(first_round_analysis)
        print(f"解析结果: 需要搜索={analysis.need_search}, 搜索查询={analysis.search_query}")
        
        # 检查是否成功解析为JSON格式
        if analysis.search_query is not None or not analysis.need_search:
            print("✓ JSON格式解析成功")
            break
        else:
            print(f"✗ JSON格式解析失败，尝试重试...")
            if attempt < max_retries - 1:
                print("等待2秒后重试...")
                import time
                time.sleep(2)
    
    if analysis is None:
        print("警告：多次重试后仍无法解析JSON格式，使用默认值")
        if image_paths:
            analysis = FirstRoundAnalysis(
                image_analysis=first_round_analysis or "无法解析图片内容",
                need_search=True,
                search_query="pubchem数据库使用与下载方式",
                search_reason="需要获取专业的化学计算方法和数据库信息"
            )
        else:
            analysis = FirstRoundAnalysis(
                image_analysis=first_round_analysis or "无法解析问题内容",
                need_search=True,
                search_query="化学溶解平衡计算方法和pH值计算工具",
                search_reason="需要获取专业的化学计算方法和数据库信息来准确计算溶解平衡和pH值"
            )
    
    # 判断是否需要搜索并执行搜索
    actual_search_results = search_results  # 使用传入的搜索结果
    summary_search = None  # 初始化搜索总结
    
    if not actual_search_results:  # 如果没有传入搜索结果，则根据第一轮结果决定是否搜索
        actual_search_results = perform_search_if_needed(query, first_round_analysis)
    
    # 如果有搜索结果，进行总结
    if actual_search_results and isinstance(actual_search_results, dict) and "prompt" in actual_search_results:
        try:
            client = OpenAI(
                api_key="sk-dkqEVEHBBbWtdmwLeyc0xyGxfcNTTHTESX5cmr4jxIh6S00M",
                base_url="https://zjuapi.com/v1"
            ) 

            api_messages = [
                {"role": "system", "content": "你是一个专业的AI总结助手"},
                {"role": "user", "content": actual_search_results["prompt"]}
            ]
            
            if use_stream:
                # 使用流式输出避免超时
                print("🔄 搜索总结使用流式输出模式...")
                summary_search = _stream_chat_completion(
                    client, 
                    api_messages, 
                    model="claude-sonnet-4-5-20250929", 
                    temperature=0.7
                )
            else:
                # 使用非流式输出
                print("📝 搜索总结使用非流式输出模式...")
                response = client.chat.completions.create(
                    model="anthropic/claude-sonnet-4.5",
                    messages=api_messages,
                    temperature=0.7
                )
                summary_search = response.choices[0].message.content
            
            print(f"\n✓ 搜索总结完成，长度: {len(summary_search)} 字符")
        except Exception as e:
            print(f"⚠ 搜索总结失败: {e}，使用原始搜索结果")
            summary_search = actual_search_results.get("search_results", str(actual_search_results))
    elif actual_search_results:
        # 如果actual_search_results是字符串，直接使用
        summary_search = str(actual_search_results)
    
    # 第二轮：科学工具生成
    stream_mode_text = "流式输出模式" if use_stream else "非流式输出模式"
    print(f"\n=== 第二轮：科学工具生成（{stream_mode_text}）===")
    second_round_conversations = create_second_round_conversation(
        query, answer, image_paths, 
        summary_search if summary_search else "[无搜索结果]", 
        analysis.get_analysis_content()
    )
    # 根据use_stream参数决定是否使用流式输出
    second_round_result = multi_turn_chat(second_round_conversations, stream=use_stream)
    
    # 合并两轮结果
    complete_result = first_round_result + second_round_result
    
    # 显示完整对话历史
    display_conversation_history(complete_result)
    
    return complete_result

def create_first_round_conversation(query: str, image_paths: List[str] = None,subfield:str = None) -> List[ConversationTurn]:
    """创建第一轮对话：图片识别和描述，并判断是否需要搜索"""
    if image_paths is None:
        image_paths = []
    elif isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # 根据是否有图片调整系统提示
    if image_paths:
        system_instruction = f"你是一个{subfield}学科专家能识别各种学科专业的图片，请对图片的内容、细节与关键信息作出描述。"
    else:
        system_instruction = "你是一个{subfield}学科专家，擅长分析各种学科专业问题，专注于判断是否需要网络搜索来获取更多信息。"
    
    first_round_system = f"""{system_instruction}

你需要判断当前问题是否需要额外的网络搜索来获取更多信息。如果需要搜索，请明确指出需要搜索的关键词和搜索目的。

重要：你必须严格按照指定的JSON格式返回结果，不要添加任何其他文字。"""
    
    # 根据是否有图片调整用户消息
    if image_paths:
        first_round_user = """
任务：
1. 识别图片中的学科领域、关键元素和专业信息
2. 描述图片的核心内容和细节特征
3. 判断是否需要搜索外部资源
4. 生成针对图片内容的优化搜索查询

图片分析要点：
- 识别学科类型（数学公式、化学结构、生物图谱、工程图纸、材料特征等）
- 提取关键符号、术语、数据
- 判断图片展示的具体问题或概念

搜索查询规则：
- 基于内容生成3-6个关键词(可以采用多语言的关键词联合搜索)
- 包含识别到的专业术语、符号、概念名称
- 添加限定词：tutorial, example, solution, 教程, 解析,详解,解读等
- 如果图片包含特定符号/公式，在查询中体现 
- 有类似像知乎/贴吧/学术论坛/公众号这样的网址可以重点查看

返回JSON格式：
{{
    "image_description": "详细描述图片内容、学科领域、关键元素和细节信息",
    "identified_concepts": ["概念1", "概念2", "概念3"],
    "problem_analysis": "分析图片展示的核心问题和学科任务",
    "need_search": true/false,
    "search_query": "基于图片内容的搜索查询（关键词形式）",
    "search_reason": "需要搜索什么类型的资源来解决图片中的问题"
}}

示例1 - 数学公式图片：
{{
    "image_description": "图片展示一个微分方程：dy/dx + p(x)y = q(x)，属于一阶线性微分方程，旁边有初始条件y(0)=1",
    "identified_concepts": ["一阶线性微分方程", "初值问题", "常微分方程"],
    "problem_analysis": "核心是求解一阶线性微分方程的初值问题，属于微积分/常微分方程领域",
    "need_search": true,
    "search_query": "first order linear differential equation solution method 或 一阶线性微分方程 求解步骤",
    "search_reason": "需要获取一阶线性微分方程的标准求解方法和步骤教程"
}}

示例2 - 化学结构式：
{{
    "image_description": "图片显示一个有机化合物结构式，包含苯环、羟基(-OH)和羧基(-COOH)，疑似水杨酸结构",
    "identified_concepts": ["有机化合物", "苯环", "羟基", "羧基", "水杨酸"],
    "problem_analysis": "核心是识别和命名有机化合物结构，属于有机化学领域的结构解析任务",
    "need_search": true,
    "search_query": "salicylic acid structure properties 或 水杨酸 化学性质 反应",
    "search_reason": "需要获取该化合物的标准命名、性质和相关反应信息"
}}

示例3 - 电路图：
{{
    "image_description": "图片展示一个RC串联电路，包含电阻R=10kΩ、电容C=100μF、电源V=5V，标注了电压和电流方向",
    "identified_concepts": ["RC电路", "串联电路", "电容充放电", "时间常数"],
    "problem_analysis": "核心是分析RC电路的充放电过程，属于电路分析领域的瞬态响应问题",
    "need_search": true,
    "search_query": "RC circuit charging discharging calculation 或 RC电路 时间常数 计算公式",
    "search_reason": "需要获取RC电路的充放电公式、时间常数计算方法和波形分析"
}}

示例4 - 生物图谱：
{{
    "image_description": "图片显示细胞有丝分裂的不同阶段示意图，包含前期、中期、后期、末期的染色体形态变化",
    "identified_concepts": ["有丝分裂", "染色体", "细胞分裂", "细胞周期"],
    "problem_analysis": "核心是理解细胞有丝分裂的各个阶段特征，属于细胞生物学领域",
    "need_search": true,
    "search_query": "mitosis stages diagram explanation 或 有丝分裂 各时期特点 图解",
    "search_reason": "需要获取有丝分裂各阶段的详细解释和特征对比"
}}

示例5 - 几何图形：
{{
    "image_description": "图片展示一个三角形ABC，标注了边长a=5, b=7, c=8，求角A的度数",
    "identified_concepts": ["三角形", "余弦定理", "解三角形"],
    "problem_analysis": "核心是利用三边长求角度，属于三角函数/解析几何领域",
    "need_search": true,
    "search_query": "law of cosines formula calculator 或 余弦定理 求角度 公式",
    "search_reason": "需要获取余弦定理的公式和计算步骤"
}}

示例6 - 图表数据：
{{
    "image_description": "图片显示一个折线图，横轴是时间(2020-2024)，纵轴是销售额，展示了5年的增长趋势",
    "identified_concepts": ["时间序列", "趋势分析", "数据可视化"],
    "problem_analysis": "核心是分析时间序列数据的增长趋势，属于数据分析/统计学领域",
    "need_search": true,
    "search_query": "time series trend analysis methods 或 时间序列 趋势分析 Python",
    "search_reason": "需要获取时间序列分析的方法和工具"
}}

关键点：
- image_description要详细具体，包含关键符号、数值、标注
- identified_concepts提取3-5个核心概念
- search_query基于识别到的专业内容生成，使用准确术语
- 如果图片内容不清晰或无法识别，在image_description中说明

现在请分析图片并生成搜索查询：
问题与图片：{query}\n{images}
""".format(query= query,images = image_paths)
    else:
        first_round_user = """分析题目并生成Google搜索查询。
任务：
1. 分析问题的核心需求、学科领域和具体任务
2. 判断是否需要搜索外部资源比如化学反应数据库，生物基因数据库等
3. 如果需要搜索，生成优化的Google搜索查询


搜索查询规则：
- 基于图片内容生成3-6个关键词
- 包含识别到的专业术语、符号、概念名称
- 添加限定词：tutorial, example, solution, 教程, 解析,详解,解读等
- 如果图片包含特定符号/公式，在查询中体现 
- 有类似像知乎/贴吧/学术论坛/公众号这样的网址可以重点查看

返回JSON格式：
{{
    "problem_analysis": "分析问题的核心需求和细分学科领域与学科任务",
    "need_search": true/false,
    "search_query": "优化后的搜索查询（关键词形式）",
    "search_reason": "为什么需要搜索，期望找到什么类型的资源"
}}

示例1 - 需要数据库：
问题：如何进行蛋白质结构预测？
{{
    "problem_analysis": "核心需求是蛋白质结构预测方法，属于生物信息学领域的结构预测任务",
    "need_search": true,
    "search_query": "protein structure database PDB open source 或 AlphaFold dataset github",
    "search_reason": "需要获取蛋白质结构数据库（如PDB）和开源预测工具的相关资源"
}}

示例2 - 需要经验方法：
问题：深度学习模型如何调参？
{{
    "problem_analysis": "核心需求是深度学习超参数优化，属于机器学习领域的模型训练任务",
    "need_search": true,
    "search_query": "深度学习 调参技巧 知乎 或 hyperparameter tuning best practices",
    "search_reason": "需要获取专家总结的调参经验和实战技巧"
}}

示例3 - 需要数据库和方法：
问题：如何分析基因表达数据？
{{
    "problem_analysis": "核心需求是基因表达数据分析，属于生物信息学领域的转录组学分析任务",
    "need_search": true,
    "search_query": "gene expression database GEO NCBI 或 RNA-seq analysis tutorial",
    "search_reason": "需要获取基因表达数据库（如GEO）和分析流程教程"
}}

示例4 - 不需要搜索：
问题：1+1等于几？
{{
    "problem_analysis": "简单的数学加法计算，属于基础算术",
    "need_search": false,
    "search_query": "",
    "search_reason": ""
}}

关键点：
- search_query必须是关键词组合（如"protein database PDB"），不要写成问句
- 可以用"或"连接多个查询策略
- 优先使用领域专业术语
- 根据资源类型选择中英文

现在请为以下问题生成搜索查询：
问题：{query}
""".format(query = query)
   
    return [
        ConversationTurn("system", first_round_system),
        ConversationTurn("user", first_round_user, images=image_paths)
    ]

def create_second_round_conversation(query: str, answer:str,image_paths: List[str] = None, search_results: str = None, first_round_result: str = None) -> List[ConversationTurn]:
    """创建第二轮对话：科学工具生成"""
    if image_paths is None:
        image_paths = []
    elif isinstance(image_paths, str):
        image_paths = [image_paths]
    
    # 读取第二轮的system prompt
    with open("../prompts/SystemPrompt_Science_Toolkit.md", 'r', encoding='utf-8') as f:
        second_round_system = f.read()
    
    # 构建第二轮的用户消息
    second_round_user_parts = [
        f"科学问题：{query}",
        f"标准答案：{answer}",
        f"图片分析结果：{first_round_result if first_round_result else '[等待第一轮分析结果]'}",
        f"搜索工具结果：{search_results if search_results else '[等待搜索工具结果]'}"
    ]

    second_round_user = "\n\n".join(second_round_user_parts)
    
    return [
        ConversationTurn("system", second_round_system),
        ConversationTurn("user", second_round_user, images=image_paths)
    ]

def load_data_file(file_path: str) -> List[Dict[str, Any]]:
    """加载数据文件，支持 JSON 和 JSONL 两种格式
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        数据列表
    """
    # 根据文件扩展名判断格式
    if file_path.endswith('.jsonl'):
        # JSONL 格式：每行一个 JSON 对象
        datasets = []
        with open(file_path, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    data = json.loads(line)
                    datasets.append(data)
                except json.JSONDecodeError as e:
                    print(f"警告：第 {line_num} 行 JSON 解析失败: {e}")
                    continue
        print(f"从 JSONL 文件加载了 {len(datasets)} 条数据")
        return datasets
    else:
        # JSON 格式：整个文件是一个 JSON 数组或对象
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            # 如果是字典，转换为列表
            if isinstance(data, dict):
                datasets = [data]
            elif isinstance(data, list):
                datasets = data
            else:
                raise ValueError(f"不支持的 JSON 格式：期望数组或对象，得到 {type(data)}")
        print(f"从 JSON 文件加载了 {len(datasets)} 条数据")
        return datasets

def main():
    """主函数 - 示例用法"""
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='科学工具生成器')
    parser.add_argument('--parser', action='store_true', help='启用代码解析功能')
    parser.add_argument('--data-file', default='./gpqa_physics_chemistry_problems_mechanics.json', help='数据文件路径（支持 JSON 和 JSONL 格式）')
    parser.add_argument('--stream', action='store_true', help='使用流式输出模式（避免长响应超时，默认使用非流式）')
    
    args = parser.parse_args()
    
    # 示例数据
    problem = {
        "id": "C005/0009",
        "question": "根据分子结构 <image>，计算五个利平斯基规则指标，并将值四舍五入到小数点后一位：分子量、LogP、氢键供体数量、氢键受体数量和可旋转键数量。请以JSON字典的形式输出，使用以下精确的键值（不包含单位）：\n\n{\n  \"分子质量\": ,\n  \"XLogP\": ,\n  \"氢键供体计数\": ,\n  \"氢键受体计数\": ,\n  \"可旋转键计数\": \n}",
        "answer": "{'分子质量': 518.7, 'XLogP': -0.7, '氢键供体计数': 7.0, '氢键受体计数': 10.0, '可旋转键计数': 11.0}",
        "images": [
            "data/images/C005_0009_3165cfeaca24fa54f61d8a43cb277f17.png"
        ],
        "metadata": {
            "qustion_type": "exact_match",
            "field": "chemistry",
            "lang": "",
            "image_urls": [
                "https://huggingface.co/datasets/Soptq/sfe/resolve/main/images/C005_0009_3165cfeaca24fa54f61d8a43cb277f17.png"
            ],
            "source_dataset": "sfe"
        }
    } 
    
    # 加载数据文件（支持 JSON 和 JSONL）
    datasets = load_data_file(args.data_file)  
    for i, data in enumerate(datasets):
        # idx = data["index"] if data.get("index") else data["id"] 
        if data.get("index"):
            idx = data["index"] 
        elif  data.get("id"):
            idx = data["id"]  
        else: 
            idx =  i 
   
        query = data["question"]
        # 兼容不同的图片字段名：image_path 或 images
        # 如果 image_path 是字符串，直接使用；如果是 images 数组，也支持
        image_paths = data.get("image_path")
        if not image_paths:
            image_paths = data.get("images")
        # 确保 image_paths 是列表格式（后续函数会处理字符串转列表）
        if image_paths is None:
            image_paths = []
        elif isinstance(image_paths, str):
            image_paths = [image_paths]
        # 如果 images 是空列表，保持为空列表
        elif isinstance(image_paths, list) and len(image_paths) == 0:
            image_paths = []
        
        answer = data["answer"] 
        # 兼容不同的子领域字段：classification_subfield 或 metadata.subfield
        subfield = data.get("classification_subfield")
        if not subfield and data.get("metadata"):
            subfield = data["metadata"].get("subfield")
        if not subfield:
            # 如果都没有，尝试从 metadata.field 获取
            if data.get("metadata"):
                subfield = data["metadata"].get("field")
            if not subfield:
                print(f"警告：第 {i+1} 条数据未找到子领域字段，使用默认值")
                subfield = "Unknown"
  

  
        # 不传入预定义的搜索结果，让系统自动判断是否需要搜索
        search_results = None  # 让系统根据第一轮结果自动决定是否搜索
        
        print("=" * 60)
        print("科学工具生成器 - 带搜索功能")
        print("=" * 60)
        print(f"问题: {query}")
        # print(f"图片: {image_paths}")
        print(f"搜索工具可用: {SEARCH_AVAILABLE}")
        print(f"Parser 功能: {'启用' if args.parser else '禁用'}")
        print(f"输出模式: {'流式' if args.stream else '非流式（默认）'}")
        
        # 生成科学工具
        result = generate_sci_tool(query, answer, image_paths, search_results, subfield, use_stream=args.stream)
        
        # 保存结果
        serializable_result = []
        for turn in result:
            turn_dict = {
                "role": turn.role,
                "content": turn.content,
                "images": turn.images,
                "metadata": turn.metadata
            }
            serializable_result.append(turn_dict) 

       
        save_path = f"../result_mid/chem_bench_tool_result_{idx}.json" 

        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=4, ensure_ascii=False)
        
        print("结果已保存到", save_path) 

        
      

if __name__ == "__main__":
    main()
