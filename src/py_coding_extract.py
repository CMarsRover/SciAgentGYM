import json 
import re


def extract_number_from_path(path: str,date='1021') -> int:
    """
    从形如 'result_mid/open_end_物理科学_204.json' 的路径里抽出 157
    """
    # 找到最后一个 '_' 和 '.json' 之间的数字
    if date in path:
        grps = path.split("_")[2]+"_"+path.split("_")[3]
        return grps
    match = re.search(r'_(\d+)\.json$', path) 
    # match_0 = re.search(r'(?<=HLE_bench_tool_result_)[0-9a-f]+(?=\.json)', path)
    # if match_0:
    #     return match_0.group(0)
    # print(match)
    if not match:
        raise ValueError("路径格式不符合预期，找不到数字")
    return int(match.group(1))

def extract_python_code(text):
    """
    抽取markdown中所有python代码块的内容
    
    Args:
        text: 包含markdown格式的文本
        
    Returns:
        list: 所有python代码块的内容列表
    """
    # 正则表达式模式：匹配 ```python 到 ``` 之间的内容
    pattern = r'```python\s*\n(.*?)\n```'
    
    # 使用 re.DOTALL 让 . 匹配换行符
    matches = re.findall(pattern, text, re.DOTALL)
    print(matches)
    return matches

def extract_python_code_with_position(text):
    """
    抽取python代码块并返回位置信息
    
    Returns:
        list: [(代码内容, 开始位置, 结束位置), ...]
    """
    pattern = r'```python\s*\n(.*?)\n```'
    
    results = []
    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1)
        start_pos = match.start()
        end_pos = match.end()
        results.append((code_content, start_pos, end_pos))
    
    return results

def extract_code_from_content(content):
    """
    从包含代码块的content字段中抽取Python代码
    
    Args:
        content: 包含```python代码块的文本内容
        
    Returns:
        list: 提取到的代码块列表
    """
    # 匹配 ```python 到 ``` 之间的内容
    pattern = r'```python\n(.*?)(?:\n```|$)'
    
    # 使用 re.DOTALL 让 . 匹配换行符
    matches = re.findall(pattern, content, re.DOTALL)
    
    return matches

def extract_code_with_filename(content):
    """
    提取代码并尝试获取文件名信息
    
    Returns:
        list: [(filename, code), ...]
    """
    results = []
    
    # 首先尝试从代码块中提取（```python ... ```格式）
    pattern = r'```python\n(.*?)(?:\n```|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for code in matches:
        filename = extract_filename_from_code(code)
        if filename is None:
            # 如果提取不到文件名，使用默认文件名
            filename = 'gpqa_physics_chemsirty.py'
        results.append((filename, code))
    
    # 如果没有找到代码块，尝试从整个内容中提取
    if not results:
        # 检查整个内容是否以 # Filename: 开头
        filename_match = re.search(r'# Filename:\s*([^\s\n\r#]+\.py)', content)
        if filename_match:
            filename = filename_match.group(1).strip()
            # 清理文件名，移除可能的注释内容
            if '(' in filename and ')' in filename:
                filename = filename.split('(')[0].strip()
            # 确保文件名以.py结尾
            if not filename.endswith('.py'):
                filename += '.py'
            results.append((filename, content))
        else:
            # 如果没有找到明确的文件名，使用默认文件名
            filename = 'extracted_code.py'
            results.append((filename, content))
    
    return results

def extract_filename_from_code(code):
    """
    从代码中提取文件名
    
    Args:
        code: Python代码字符串
        
    Returns:
        str: 提取的文件名
    """
    # 尝试从代码中提取文件名，支持多种格式
    filename_match = re.search(r'# Filename:\s*([^\s\n\r#]+\.py)', code)
    if not filename_match:
        # 也尝试匹配其他可能的格式
        filename_match = re.search(r'# 文件名:\s*([^\s\n\r#]+\.py)', code)
    if not filename_match:
        # 尝试匹配文件开头的注释（更严格的模式）
        filename_match = re.search(r'#\s*([a-zA-Z_][a-zA-Z0-9_]*\.py)', code)
    
    filename = filename_match.group(1).strip() if filename_match else None
    
    # 如果提取到了文件名，进行清理
    if filename:
        # 清理文件名，移除可能的注释内容
        if '(' in filename and ')' in filename:
            # 如果文件名包含括号，只取括号前的部分
            filename = filename.split('(')[0].strip()
        
        # 确保文件名以.py结尾
        if not filename.endswith('.py'):
            filename += '.py'
    
    return filename

def clean_extracted_code(code):
    """
    清理提取的代码，移除不完整的行
    """
    lines = code.split('\n')
    
    # 移除最后不完整的行（比如函数定义没有结束）
    cleaned_lines = []
    for line in lines:
        # 如果行以冒号结尾但后面没有内容，可能是截断的
        if line.strip().endswith(':') and line.strip().startswith('def '):
            # 检查下一行是否有缩进内容
            continue
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def process_response_content(response_text):
    """
    处理完整的响应内容，提取所有代码块
    
    Args:
        response_text: 完整的响应文本或JSON字符串
        
    Returns:
        dict: 处理结果
    """
    try:
        # 如果是JSON字符串，先解析
        if response_text.strip().startswith('{'):
            data = json.loads(response_text)
            content = data.get('content', response_text)
        else:
            content = response_text
        
        # 提取代码
        codes = extract_code_from_content(content)
        codes_with_filename = extract_code_with_filename(content)
        
        result = {
            'total_code_blocks': len(codes),
            'codes': codes,
            'codes_with_filename': codes_with_filename,
            'cleaned_codes': [clean_extracted_code(code) for code in codes]
        }
        
        return result
        
    except Exception as e:
        return {'error': f'处理失败: {e}'}

# 使用示例
def main():
    import pathlib 
    import os
    import argparse
    mids_path = list(pathlib.Path('../result_mid').rglob('*.json'))   
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='科学工具生成器')
    parser.add_argument('--parser', action='store_true', help='启用代码解析功能')
    parser.add_argument('--output-file', default='./extracted_tools_1028', help='工具文件保存路径')
    
    args = parser.parse_args()
    
    
    for jsn in mids_path:
        fl_name = jsn.name 
    
        fl_keys = fl_name.split("_") 
    
        if  fl_keys[0] ==  "GPQA":
          
            with open(os.path.join("../result_mid",fl_name),"r",encoding='utf-8')as f: 
                context = json.load(f) 
                
            idx = extract_number_from_path(os.path.join("../result_mid",fl_name))
            py_fl = context[-1] 
            save_extracted_code(py_fl["content"],output_dir=args.output_file,idx=idx)

  
def test_main():
    # idx = extract_number_from_path("result_mid/open_end_物理科学_218.json")

    idx = extract_number_from_path("result_mid/open_end_物理科学_204.json")
   
    with open("extracted_test_code/open_end_物理科学_204.json","r",encoding="utf-8")as f: 
        context = json.load(f) 

    py_fl = context[0]
    
    save_extracted_code(py_fl["content"],output_dir="./test",idx = idx)

# 保存代码到文件
def save_extracted_code(content, output_dir="./extracted_code_1023",stage=1,idx=0):
    """将提取的代码保存到文件"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    codes_with_filename = extract_code_with_filename(content)
    print(f"提取到 {len(codes_with_filename)} 个代码块")

    for i, (filename, code) in enumerate(codes_with_filename):
        # 确保文件名安全
        safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
        if not safe_filename.endswith('.py'):
            safe_filename += '.py'
        if stage == 2:
            filepath = os.path.join(output_dir, "tst"+str(idx)+safe_filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code) 
        else:
            fl_name = safe_filename.split(".py")[0]
            filepath = os.path.join(output_dir, fl_name+"_claude_"+str(idx)+".py")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code) 
        
        print(f"已保存: {filepath}")

if __name__ == "__main__":

    main() 

    