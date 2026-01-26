import os
import sys
import subprocess
import base64
import mimetypes
import json
import re
import pdb
from typing import Dict, List, Optional, Tuple, Union 
from pydantic import BaseModel, Field
from openai import OpenAI 

def simple_chat(message, model="claude-sonnet-4-5-20250929", images=None):
    """使用OpenAI客户端进行简单聊天，支持图片"""
    # 创建客户端
    client = OpenAI(
        api_key="sk-dkqEVEHBBbWtdmwLeyc0xyGxfcNTTHTESX5cmr4jxIh6S00M",
        base_url="https://zjuapi.com/v1"
    )
    try:
        message_content = []
        # 添加文本内容
        if message:
            message_content.append({
                "type": "text",
                "text": message
            })
        
        # 添加图片内容
        if images:
            # 确保images是列表格式
            if isinstance(images, str):
                images = [images]
            elif images is None:
                images = []
            
            for image_path in images:
                if image_path:  # 确保图片路径不为空
                    try:
                        # 编码图片为base64
                        with open(image_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                        
                        # 获取图片MIME类型
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
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message_content}
            ],
            temperature=0.2,
            max_tokens=64000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"请求异常: {e}"

def run_python_file(
    file_path: str,
    args: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    input_text: Optional[str] = None,
) -> Tuple[int, str, str]:
    """执行指定的 Python 脚本文件并返回结果。
    轻量脚本执行工具
    提供函数以子进程方式执行指定的 Python 脚本文件，并返回退出码、标准输出与错误输出。

    示例：
        from script_runner import run_python_file
        code, out, err = run_python_file("/path/to/script.py", args=["--flag", "123"], timeout=30)
        if code == 0:
            print(out)
        else:
            print(err)


    参数:
        file_path: 要执行的脚本绝对路径或相对路径。
        args: 传递给脚本的命令行参数列表（不包含 python 与脚本本身）。
        cwd: 作为脚本运行时的工作目录；不传则使用当前进程目录。
        env: 额外或覆盖的环境变量字典；在当前环境基础上合并。
        timeout: 超时时间（秒）。
        input_text: 通过标准输入传入的文本内容；默认为空。

    返回:
        (returncode, stdout, stderr)
    """

    if not file_path:
        raise ValueError("file_path 不能为空")

    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到脚本文件: {file_path}")

    cmd: List[str] = [sys.executable, file_path]
    if args:
        cmd.extend(args)

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=merged_env,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # 超时情况下返回码约定为 124（对齐常见 shell 约定），并返回已捕获输出
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
        return 124, stdout, (stderr or "Execution timed out")

    return result.returncode, result.stdout, result.stderr 

def llm_judge(gt,llm_return):



    pattern = r'={60}\n场景1：[^\n]*\n={60}[\s\S]*?(?=\n={60}\n场景2：|$)'
    match = re.search(pattern, llm_return, re.DOTALL) 
    print("正则匹配结果:", match is not None)
    if match:
        scenario1_result = match.group(0)
        print("场景一内容长度:", len(scenario1_result))
        print("场景一内容前100字符:", scenario1_result[:100])
    else:
        print("未能匹配到场景一，原始输出前500字符:")
        print(out[:500])
        exit()



    judge_prompt = f""" 
    请判断调用工具的结果和gt是否一致，只需要给出True/False
    INPUTS：
       运行结果：{scenario1_result}
       标准答案：{gt}
    OUTPUT：
        请回答运行结果与标准答案是都一致
    """ 
    final_output = simple_chat(message=judge_prompt)  
    return final_output


if __name__ == "__main__":
    import pathlib 
    py_paths = list(pathlib.Path('/Users/yangyajie/Desktop/code/sci_tool_env/extracted_tools_1113/tools').rglob('*.py'))   
    
    for i, fl in enumerate(py_paths):
        print(f"\n{'='*60}")
        print(f"正在处理第 {i+1}/{len(py_paths)} 个文件: {fl}")
        print(f"{'='*60}")
        
        code, out, err = run_python_file(fl, args=["--help"])
        
        print(f"退出码: {code}")
        print(f"标准输出:\n{out}")
        if err:
            print(f"错误输出:\n{err}")
        
        # 设置断点，让你可以查看结果
        print(f"\n文件 {fl} 处理完成。按 'c' 继续下一个文件，或使用其他 pdb 命令进行调试。")
        pdb.set_trace()  


