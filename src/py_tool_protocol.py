from sci_tool_generator import ConversationTurn,multi_turn_chat
import os 
import re
import json 
import pathlib 
import argparse

def create_protocol(py_code:str,py_name:str): 
    with open("../prompts/Prompt_protocol.md", 'r', encoding='utf-8') as f:
        protocol_content = f.read()   
    return[
            ConversationTurn("user", f"{protocol_content}\n文件名:\n{py_name}\n工具代码:\n{py_code}"),
        ] 

def main(): 
    parser = argparse.ArgumentParser(description='科学工具生成器')
    parser.add_argument('--parser', action='store_true', help='启用代码解析功能')
    parser.add_argument('--tool-file', default='../extracted_tools_1024', help='数据文件路径')
    
    args = parser.parse_args()
    
    tools_path = list(pathlib.Path(args.tool_file).rglob('*.py'))  

    for py in tools_path:
        fl_name = py.name 
        print(f"📁 处理文件: {py}")
        print(f"文件名: {fl_name}")
      
        try:
            with open(py, "r", encoding='utf-8') as f: 
                code = f.read()
        except Exception as e:
            print(f"❌ 读取文件失败 {py}: {e}")
            continue  
        save_name  = fl_name.split('.py')[0]
        conversation = create_protocol(code,fl_name)  
        print(f"📝 开始处理文件: {fl_name}")
        print(f"对话消息数量: {len(conversation)}")
        
        try:
            result = multi_turn_chat(conversation) 
            print(f"✅ 对话完成，收到 {len(result)} 条消息")
            
            # 检查结果是否包含助手回复
            has_assistant = any(turn.role == "assistant" for turn in result)
            if not has_assistant:
                print(f"⚠️ 警告：对话结果中没有助手回复！")
                print(f"结果中的消息角色: {[turn.role for turn in result]}")
                continue
            
            # 将ConversationTurn对象转换为可序列化的字典格式
            serializable_result = []
            for turn in result:
                turn_dict = {
                    "role": turn.role,
                    "content": turn.content,
                    "images": turn.images,
                    "metadata": turn.metadata
                }
                serializable_result.append(turn_dict) 
            
            # 获取最后一个助手回复
            assistant_messages = [msg for msg in serializable_result if msg["role"] == "assistant"]
            if not assistant_messages:
                print(f"❌ 错误：没有找到助手回复，跳过文件 {fl_name}")
                continue
            
            last_assistant_content = assistant_messages[-1]["content"]
            print(f"📄 最后一条助手回复长度: {len(last_assistant_content)} 字符")
            
            # 提取JSON
            final_json = extract_json_from_markdown(last_assistant_content)
            
            if final_json is None:
                print(f"⚠️ 警告：无法从助手回复中提取JSON，跳过文件 {fl_name}")
                print(f"助手回复前500字符: {last_assistant_content[:500]}")
                continue
            
            # 保存结果
            with open(f"../extracted_tools_1118/protocols/{save_name}_protocol.json","w",encoding="utf-8")as f: 
                json.dump(final_json,f,indent=4,ensure_ascii=False)
                print(f"✅ 保存成功: {save_name}")
                
        except Exception as e:
            import traceback
            print(f"❌ 处理文件 {fl_name} 时出错: {e}")
            print(f"错误详情:")
            traceback.print_exc()
            continue

def extract_json_from_markdown(text):
    """
    从包含 ```json ... ``` 的文本中提取 JSON 内容。
    返回提取到的 JSON 字符串，若成功则解析为 Python 对象；否则返回 None。
    """
    if not text:
        print("输入文本为空")
        return None
        
    # 使用正则表达式匹配 ```json 和 ``` 之间的内容（支持多行）
    pattern = r'```json\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL 让 . 匹配换行符

    if match:
        json_str = match.group(1).strip()  # 提取并去除首尾空白
        try:
            return json.loads(json_str)  # 解析为 Python 对象（dict/list 等）
        except json.JSONDecodeError as e:
            print(f"提取到的字符串不是合法 JSON: {e}")
            print(f"原始字符串: {json_str[:200]}...")  # 只显示前200个字符
            return None
    else:
        # 尝试查找其他可能的JSON格式
        # 检查是否整个文本就是JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
            
        # 查找可能的JSON数组或对象
        json_patterns = [
            r'\[.*\]',  # 数组
            r'\{.*\}',  # 对象
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
                    
        print("未找到有效的JSON格式")
        return None

if __name__ == "__main__":
    main()
