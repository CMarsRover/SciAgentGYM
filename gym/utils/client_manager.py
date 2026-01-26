#!/usr/bin/env python3
"""
多模型客户端管理器
支持OpenAI、Claude、DeepSeek、Qwen、Gemini、Kimi等多种模型
"""

import json
import os
from typing import Dict, List, Optional, Any
from gym.config.config import SUPPORTED_MODELS, DEFAULT_MODEL, TEMPERATURE


class UnifiedClient:
    """统一的多模型客户端"""
    
    def __init__(self, model_name: str = None):
        initial_name = model_name or DEFAULT_MODEL
        if initial_name not in SUPPORTED_MODELS:
            # 严格模式：未知模型直接报错，避免误用其他模型的 API
            raise ValueError(f"未识别的模型名: {initial_name}，请在 SUPPORTED_MODELS 中选择有效模型")
        self.model_name = initial_name
        self.model_config = SUPPORTED_MODELS.get(self.model_name, {})
        self.provider = self.model_config.get("provider", "openai")
        # 仅使用 SUPPORTED_MODELS 中为该模型配置的 base_url / api_key（严格模式）
        self.api_base_url = self.model_config.get("api_base_url")
        self.api_key = self.model_config.get("api_key")
        self._client = None
        
        # 初始化对应的客户端
        self._init_client()
    
    def _init_client(self):
        """根据提供商初始化对应的客户端"""
        if self.provider == "openai":
            self._init_openai_client()
        elif self.provider == "anthropic":
            self._init_anthropic_client()
        elif self.provider == "deepseek":
            self._init_deepseek_client()
        elif self.provider == "qwen":
            self._init_qwen_client()
        elif self.provider == "google":
            self._init_google_client()
        elif self.provider == "moonshotai":
            self._init_moonshot_client()
        elif self.provider == "zhipuai" or self.provider == "glm":
            self._init_zhipuai_client()
        else:
            # 默认使用OpenAI兼容接口
            self._init_openai_client()
    
    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
    
    def _init_anthropic_client(self):
        """初始化Anthropic客户端"""
        try:
            from openai import OpenAI
            # 使用OpenAI兼容接口访问Claude
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
    
    def _init_deepseek_client(self):
        """初始化DeepSeek客户端"""
        try:
            from openai import OpenAI
            # 使用OpenAI兼容接口访问DeepSeek
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
    
    def _init_qwen_client(self):
        """初始化Qwen客户端"""
        try:
            from openai import OpenAI
            # 使用OpenAI兼容接口访问Qwen
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
    
    def _init_google_client(self):
        """初始化Google Gemini客户端"""
        try:
            from openai import OpenAI
            # 使用OpenAI兼容接口访问Gemini
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
    
    def _init_moonshot_client(self):
        """初始化Moonshot(Kimi)客户端"""
        try:
            from openai import OpenAI
            # 使用OpenAI兼容接口访问Kimi
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
    
    def _init_zhipuai_client(self):
        """初始化智谱AI(GLM)客户端"""
        try:
            from zhipuai import ZhipuAI
            # 检查 api_key 是否为空或格式不正确
            if not self.api_key or not isinstance(self.api_key, str) or len(self.api_key.strip()) == 0:
                raise ValueError(f"GLM 模型 {self.model_name} 的 api_key 为空或格式不正确")
            # 移除可能的空格
            api_key_clean = self.api_key.strip()
            
            # 如果提供了 base_url，可能是使用代理服务，需要使用 OpenAI 兼容接口
            if self.api_base_url:
                print(f"⚠️  检测到 base_url，GLM 将使用代理服务: {self.api_base_url}")
                try:
                    from openai import OpenAI
                    # 使用 OpenAI 兼容接口访问代理服务
                    self._client = OpenAI(api_key=api_key_clean, base_url=self.api_base_url)
                    print(f"✅ GLM 客户端初始化成功（使用代理），模型: {self.model_name}")
                except ImportError:
                    raise ImportError("使用代理服务需要安装 openai 包: pip install openai")
            else:
                # 使用官方智谱AI SDK
                self._client = ZhipuAI(api_key=api_key_clean)
                print(f"✅ GLM 客户端初始化成功（官方SDK），模型: {self.model_name}")
        except ImportError:
            raise ImportError("请安装 zhipuai 包: pip install zhipuai")
        except Exception as e:
            print(f"❌ GLM 客户端初始化失败: {e}")
            raise
    
    def chat_completions_create(self, 
                               messages: List[Dict[str, Any]], 
                               tools: Optional[List[Dict]] = None,
                               tool_choice: str = "auto",
                               parallel_tool_calls: bool = True,
                               temperature: float = None,
                               max_tokens: int = None,
                               timeout: int = 600,
                               **kwargs) -> Any:
        """
        统一的对话完成接口
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            tool_choice: 工具选择策略
            parallel_tool_calls: 是否允许并行调用工具
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 请求超时时间（秒），默认120秒
            **kwargs: 其他参数
        
        Returns:
            响应对象
        """
        if not self._client:
            raise ValueError(f"客户端未初始化: {self.provider}")
        # 快速检查必要配置，给出更清晰提示
        if not self.api_key:
            raise RuntimeError(f"模型 {self.model_name} 缺少 api_key，请在 gym/config/config.py 的 SUPPORTED_MODELS['{self.model_name}'] 中配置")
        # GLM/ZhipuAI 使用官方SDK时不需要 api_base_url，但使用代理时需要
        if self.provider not in ("zhipuai", "glm") and not self.api_base_url:
            raise RuntimeError(f"模型 {self.model_name} 缺少 api_base_url，请在 gym/config/config.py 的 SUPPORTED_MODELS['{self.model_name}'] 中配置")
        
        # 使用配置中的参数作为默认值
        if temperature is None:
            temperature = TEMPERATURE
        # if max_tokens is None:
        #     max_tokens = self.model_config.get("max_tokens", 4096)
        
        # 获取实际的模型名称
        actual_model_name = self.model_config.get("model_name", self.model_name)
        
        # 构建参数
        params = {
            "model": actual_model_name,
            "messages": messages,
            "temperature": temperature,
            # "max_tokens": max_tokens,
            "timeout": timeout,
            # "reasoning_effort": "high",     # 开启高阶推理
            # "verbosity": "high",        # 输出更详细
            **kwargs
        }
        
        # 只有在有 tools 时才添加 tool_choice 和 parallel_tool_calls
        if tools is not None:
            params["tools"] = tools
            # GLM/ZhipuAI 可能不支持某些参数，需要特殊处理
            if self.provider not in ("zhipuai", "glm"):
                params["tool_choice"] = tool_choice
                params["parallel_tool_calls"] = parallel_tool_calls
            else:
                # GLM 使用 tool_choice 但可能不支持 parallel_tool_calls
                # 注意：智谱AI可能使用 "auto" 或 "required" 作为 tool_choice
                if tool_choice == "auto":
                    params["tool_choice"] = "auto"
                elif tool_choice == "required":
                    params["tool_choice"] = "required"
                else:
                    params["tool_choice"] = "auto"  # 默认使用 auto
        
        try:
            # GLM/ZhipuAI 的调用方式与 OpenAI 兼容
            # 添加调试信息（仅在 GLM 时）
            if self.provider in ("zhipuai", "glm"):
                print(f"🔍 GLM 调用参数: model={actual_model_name}, messages数量={len(messages)}, tools数量={len(tools) if tools else 0}")
                # 检查 API key 的前几个字符（用于调试，不完整显示）
                if self.api_key:
                    key_preview = self.api_key[:10] + "..." if len(self.api_key) > 10 else self.api_key
                    print(f"🔑 API Key 预览: {key_preview}")
            
            return self._client.chat.completions.create(**params)
        except Exception as e:
            # 为常见连接错误提供更友好的提示
            msg = str(e)
            error_code = None
            # 尝试提取错误代码
            if "401" in msg:
                error_code = "401"
                print(f"❌ 调用模型 {self.model_name} 失败: 401 认证错误")
                print(f"   可能的原因：")
                print(f"   1. API Key 已过期或无效")
                print(f"   2. API Key 格式不正确（应以 'sk-' 开头）")
                print(f"   3. API Key 没有调用该模型的权限")
                print(f"   4. 请检查配置中的 api_key 是否正确")
                if self.api_key:
                    key_preview = self.api_key[:10] + "..." if len(self.api_key) > 10 else self.api_key
                    print(f"   当前 API Key 预览: {key_preview} (长度: {len(self.api_key)})")
            elif "Connection" in msg or "SSLError" in msg or "Timeout" in msg:
                print(f"调用模型 {self.model_name} 失败: 连接错误（请检查 base_url 与网络连通性）。当前 base_url='{self.api_base_url or '默认(官方)'}'")
            else:
                print(f"调用模型 {self.model_name} 失败: {e}")
            raise


class ClientManager:
    """客户端管理器"""
    
    def __init__(self):
        self._clients = {}
    
    def get_client(self, model_name: str = None) -> UnifiedClient:
        """
        获取指定模型的客户端
        
        Args:
            model_name: 模型名称，如果为None则使用默认模型
        
        Returns:
            UnifiedClient实例
        """
        model_name = model_name or DEFAULT_MODEL
        
        if model_name not in self._clients:
            self._clients[model_name] = UnifiedClient(model_name)
        
        return self._clients[model_name]
    
    def list_supported_models(self) -> List[str]:
        """列出所有支持的模型，排除仅用于评测的模型"""
        return [model_name for model_name, config in SUPPORTED_MODELS.items()
                if not config.get("evaluation_only", False)]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        return SUPPORTED_MODELS.get(model_name, {})
    
    def test_model(self, model_name: str) -> bool:
        """测试模型是否可用"""
        try:
            client = self.get_client(model_name)
            # 对于 GLM，先检查 API key 格式
            if client.provider in ("zhipuai", "glm"):
                api_key = client.api_key
                if not api_key or len(api_key.strip()) == 0:
                    print(f"❌ GLM 模型 {model_name} 的 API key 为空")
                    return False
                # 智谱AI的 API key 通常以特定格式开头
                if not api_key.startswith(("sk-", "zhipuai-")):
                    print(f"⚠️ 警告: GLM API key 格式可能不正确（应以 'sk-' 或 'zhipuai-' 开头）")
                    print(f"   当前 API key 前缀: {api_key[:10]}...")
            
            # 发送一个简单的测试消息
            response = client.chat_completions_create(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print(f"✅ 模型 {model_name} 测试成功")
            return True
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                print(f"❌ 模型 {model_name} 测试失败: 401 认证错误")
                if client.provider in ("zhipuai", "glm"):
                    print(f"   请检查：")
                    print(f"   1. API key 是否有效（可在智谱AI开放平台查看）")
                    print(f"   2. API key 是否有调用 {model_name} 的权限")
                    print(f"   3. API key 是否已过期")
                    if client.api_key:
                        key_preview = client.api_key[:15] + "..." if len(client.api_key) > 15 else client.api_key
                        print(f"   当前 API key 预览: {key_preview}")
            else:
                print(f"模型 {model_name} 测试失败: {e}")
            return False


# 全局客户端管理器实例
client_manager = ClientManager()

# 便捷函数
def get_client(model_name: str = None) -> UnifiedClient:
    """获取客户端的便捷函数"""
    return client_manager.get_client(model_name)

def list_models() -> List[str]:
    """列出所有支持的模型"""
    return client_manager.list_supported_models()

def test_all_models() -> Dict[str, bool]:
    """测试所有模型的可用性"""
    results = {}
    for model_name in list_models():
        results[model_name] = client_manager.test_model(model_name)
    return results


if __name__ == "__main__":
    # 测试客户端管理器
    print("=== 多模型客户端管理器测试 ===")
    print(f"支持的模型: {list_models()}")
    
    # 测试默认模型
    print(f"\n测试默认模型: {DEFAULT_MODEL}")
    try:
        client = get_client()
        print(f"默认模型客户端初始化成功: {client.model_name}")
    except Exception as e:
        print(f"默认模型客户端初始化失败: {e}")
    
    # 测试所有模型的可用性
    # print("\n测试所有模型的可用性...")
    # test_results = test_all_models()
    # for model, is_available in test_results.items():
    #     status = "✓" if is_available else "✗"
    #     print(f"{status} {model}")
