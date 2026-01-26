#!/usr/bin/env python3
"""
MCP服务器配置文件
本文件为统一配置源，testallforfailed 也会引用此处内容。
"""

# 通用API配置
# API_BASE_URL = 'https://zjuapi.com/v1'
# API_KEY = 'sk-vdYwRuYQ1C0Lhpr4vEo2TmgoE30oYdubJxR18makv5oPpsXV'
# API_BASE_URL = 'https://gptgod.cloud/v1'
# API_KEY = 'sk-KLtFkFghvfu1892t24E1D269Fc9c404583BfFf6fE6Eb4d7a'

# API_BASE_URL = 'https://www.dmxapi.cn/v1'
# API_KEY = 'sk-hKXNDcdR26CuaQDKp0ooBBvKtzZNGlQMunkqAD5njcF7zMAv'

# MutliSciTool专用
API_BASE_URL = ''
API_KEY = ''

# 兼容性配置
OPENAI_API_KEY = API_KEY  # 保持向后兼容

# 支持的模型配置
SUPPORTED_MODELS = {
    # # Claude 系列 - 使用Anthropic专用API
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "model_name": "claude-sonnet-4-20250514",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：Anthropic专用API
        "api_key": "sk-bhcvvaKeyQyguQ0dMUxjXUHJ3LtZPvASsLx9YnXtLdNhwD0R"  # 需要替换为实际的Anthropic API密钥
    },
    
    # "claude-haiku-4-5-20251001-thinking": {
    #     "provider": "anthropic",
    #     "model_name": "claude-haiku-4-5-20251001-thinking",
    #     "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：Anthropic专用API
    #     "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的Anthropic API密钥
    # },

    
    # # # # GPT 系列 - 使用OpenAI专用API
    "gpt-4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：OpenAI专用API
        "api_key": "sk-MrzXpVXZzWoqXZ1976S2ZcgZXoLFQxqS2M3Uk9O8ph1WdDmH"  # 需要替换为实际的OpenAI API密钥
    },
    
    "gpt-5": {
        "provider": "openai",
        "model_name": "gpt-5",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：OpenAI专用API
        "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的OpenAI API密钥
    },
    
    # # # Qwen 系列 - 使用通义千问专用API
    "Qwen/Qwen3-VL-235B-A22B-Thinking": {
        "provider": "qwen",
        "model_name": "qwen3-vl-235b-a22b-thinking",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：OpenAI专用API
        "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的OpenAI API密钥
    },
    
    "Qwen/Qwen3-VL-32B-Thinking": {
        "provider": "qwen",
        "model_name": "qwen3-vl-32b-thinking",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：OpenAI专用API
        "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的OpenAI API密钥
    },
    
    "qwen3-vl-8b-thinking": {
        "provider": "qwen",
        "model_name": "qwen3-vl-8b-thinking",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：OpenAI专用API
        "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的OpenAI API密钥
    },

    "Qwen/Qwen3-VL-30B-A3B-Thinking": {
        "provider": "qwen",
        "model_name": "qwen3-vl-30b-a3b-thinking",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：OpenAI专用API
        "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的OpenAI API密钥
    },


    # # Gemini 系列 - 使用Google专用API
    # # # Gemini 系列 - 使用Google专用API
    "gemini-2.5-pro-thinking-2048": {
        "provider": "google",
        "model_name": "gemini-2.5-pro-thinking-2048",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：Google Gemini专用API
        "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的Google API密钥
    },
    
    "gemini-2.5-pro": {
        "provider": "google",
        "model_name": "gemini-2.5-pro",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：Google Gemini专用API
        "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的Google API密钥
    },
    
    "Gemini-2.5-Flash,": {
        "provider": "google",
        "model_name": "Gemini-2.5-Flash,",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：Google Gemini专用API
        "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的Google API密钥
    },
    
    # # # # Kimi 系列 - 使用Moonshot专用API
    # "kimi-k2-thinking": {
    #     "provider": "moonshot",
    #     "model_name": "kimi-k2-thinking",
    #     "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：Moonshot专用API
    #     "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1"  # 需要替换为实际的Moonshot API密钥
    # },

    "microsoft-phi-4-multimodal-instruct":{
        "provider": "mistral",
        "model_name" : "microsoft/phi-4-multimodal-instruct",
        "api_base_url": "http://35.220.164.252:3888/v1",
        "api_key":"sk-MrzXpVXZzWoqXZ1976S2ZcgZXoLFQxqS2M3Uk9O8ph1WdDmH"
    },
    "mistrala-pixtral-12b" :{
        "provider": "mistral",
        "model_name" : "mistralai/pixtral-12b",
        "api_base_url": "http://35.220.164.252:3888/v1",
        "api_key":"sk-MrzXpVXZzWoqXZ1976S2ZcgZXoLFQxqS2M3Uk9O8ph1WdDmH"
    },
    
    # GLM/智谱AI 系列 - 使用智谱AI专用API
    # 注意：GLM 只需要 api_key，不需要 api_base_url
    "glm-4": {
        "provider": "zhipuai",
        "model_name": "glm-4",
        "api_key": ""  # 需要替换为实际的智谱AI API密钥
    },
    "glm-4-plus": {
        "provider": "zhipuai",
        "model_name": "glm-4-plus",
        "api_key": ""  # 需要替换为实际的智谱AI API密钥
    },
    "glm-4-flash": {
        "provider": "zhipuai",
        "model_name": "glm-4-flash",
        "api_key": ""  # 需要替换为实际的智谱AI API密钥
    },
    "glm-4.6v":{
        "provider": "zhipuai",
        "model_name": "glm-4.6v",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 如果使用代理服务，添加此配置
        "api_key": "sk-MrzXpVXZzWoqXZ1976S2ZcgZXoLFQxqS2M3Uk9O8ph1WdDmH"
    },
    # Judge模型配置 - 仅用于评测，不参与问答
    "gpt-4.1": {
        "provider": "openai",
        "model_name": "gpt-4.1",
        "api_base_url": "http://35.220.164.252:3888/v1",  # 示例：OpenAI专用API
        "api_key": "sk-keAtFj3FEN7nlGCBPc93Hs4IeM0bheDiEpprl33CwjleWaK1",  # 需要替换为实际的OpenAI API密钥
        "evaluation_only": True  # 标记此模型仅用于评测
    }
    
    # # Judge模型配置 - 仅用于评测，不参与问答
    # "gpt-4.1": {
    #     "provider": "openai",
    #     "model_name": "gpt-4.1",
    #     "api_base_url": "https://zjuapi.com/v1",  # 示例：OpenAI专用API
    #     "api_key": "sk-Ko9d2LvgqeuqBouaW4cO0CJcJMHbqiOJGmNRhA9my14IGQtp"  # 需要替换为实际的OpenAI API密钥
    # }
}

# AI模型配置
JUDGE_MODEL = "gpt-4.1"
DEFAULT_MODEL = "gpt-5"  # 默认使用的模型
TEMPERATURE = 0.7         # 模型温度参数
TOOL_TRACE_SUFFIX = "_react"  # 工具模式追加的trace目录后缀

# 服务器配置
MAX_ITERATIONS = 25  # 最大工具调用轮数
MAX_DEPTH = 10      # 最大递归深度
TIMEOUT = 30        # 代码执行超时时间
COMMAND_TIMEOUT = 60  # 命令执行超时时间 

# 测试配置
ENABLE_DEBUG = True  # 启用调试输出
SAVE_PLOTS = True   # 保存生成的图表
PASS_AT_K = 3       # pass@k评估中的k值,表示连续测试k次

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
 
