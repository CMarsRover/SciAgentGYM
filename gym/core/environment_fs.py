"""
统一的环境文件系统管理模块

提供统一的中介结果文件管理，避免工具中使用绝对路径导致文件存放位置混乱。

所有中间结果统一存放在 gym/mid_result/ 目录下，按领域和题目ID细分。
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime


class EnvironmentFileSystem:
    """统一的环境文件系统管理器"""
    
    # 基础目录
    BASE_DIR: Path = Path(__file__).resolve().parent.parent / "mid_result"
    
    # 支持的领域列表
    SUPPORTED_DOMAINS = {
        "physics", "chemistry", "materials", "astronomy", "geography",
        "structural_biology", "molecular_biology", "quantum_physics",
        "life_science", "earth_science", "computer_science"
    }
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        初始化环境文件系统管理器
        
        Args:
            base_dir: 基础目录路径，默认为 gym/mid_result
        """
        if base_dir is None:
            base_dir = self.BASE_DIR
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_domain_dir(self, domain: str, case_id: Optional[str] = None) -> Path:
        """
        获取领域目录路径
        
        Args:
            domain: 领域名称（如 "structural_biology", "physics"）
            case_id: 可选的题目ID，用于进一步细分目录
        
        Returns:
            Path: 领域目录路径
        
        Raises:
            ValueError: 如果领域名称不在支持列表中
        """
        # 规范化领域名称（支持下划线和连字符）
        domain = domain.lower().replace("-", "_")
        
        if domain not in self.SUPPORTED_DOMAINS:
            # 如果不在支持列表中，仍然允许使用，但给出警告
            # 这样可以支持未来新增的领域
            pass
        
        domain_dir = self.base_dir / domain
        
        # 如果提供了 case_id，创建子目录
        if case_id:
            case_id = str(case_id).strip()
            # 清理 case_id，移除可能导致路径问题的字符
            case_id = case_id.replace("/", "_").replace("\\", "_")
            domain_dir = domain_dir / case_id
        
        domain_dir.mkdir(parents=True, exist_ok=True)
        return domain_dir
    
    def save_result(
        self,
        domain: str,
        filename: str,
        data: Any,
        case_id: Optional[str] = None,
        format: str = "json",
        **kwargs
    ) -> Dict[str, Any]:
        """
        保存中间结果到文件
        
        Args:
            domain: 领域名称
            filename: 文件名（不含扩展名）
            data: 要保存的数据
            case_id: 可选的题目ID
            format: 文件格式，支持 "json", "pickle", "txt"
            **kwargs: 其他参数
                - indent: JSON 缩进（默认 2）
                - ensure_ascii: JSON 是否确保 ASCII（默认 False）
        
        Returns:
            dict: 包含保存结果的字典
                {
                    "success": bool,
                    "filepath": str,
                    "size": int,
                    "error": str (如果失败)
                }
        """
        try:
            domain_dir = self.get_domain_dir(domain, case_id)
            
            # 清理文件名
            filename = str(filename).strip()
            filename = filename.replace("/", "_").replace("\\", "_")
            
            # 根据格式确定文件扩展名和保存方式
            if format.lower() == "json":
                filepath = domain_dir / f"{filename}.json"
                indent = kwargs.get("indent", 2)
                ensure_ascii = kwargs.get("ensure_ascii", False)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
            
            elif format.lower() == "pickle":
                filepath = domain_dir / f"{filename}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
            
            elif format.lower() == "txt":
                filepath = domain_dir / f"{filename}.txt"
                with open(filepath, "w", encoding="utf-8") as f:
                    if isinstance(data, str):
                        f.write(data)
                    else:
                        f.write(str(data))
            
            else:
                return {
                    "success": False,
                    "filepath": None,
                    "size": 0,
                    "error": f"不支持的格式: {format}，支持: json, pickle, txt"
                }
            
            size = filepath.stat().st_size
            
            return {
                "success": True,
                "filepath": str(filepath),
                "size": size,
                "error": None
            }
        
        except Exception as e:
            return {
                "success": False,
                "filepath": None,
                "size": 0,
                "error": str(e)
            }
    
    def load_result(
        self,
        domain: str,
        filename: str,
        case_id: Optional[str] = None,
        format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从文件加载中间结果
        
        Args:
            domain: 领域名称
            filename: 文件名（不含扩展名）
            case_id: 可选的题目ID
            format: 文件格式，如果为 None 则自动检测
        
        Returns:
            dict: 包含加载结果的字典
                {
                    "success": bool,
                    "data": Any,
                    "filepath": str,
                    "error": str (如果失败)
                }
        """
        try:
            domain_dir = self.get_domain_dir(domain, case_id)
            
            # 清理文件名
            filename = str(filename).strip()
            filename = filename.replace("/", "_").replace("\\", "_")
            
            # 如果未指定格式，尝试自动检测
            if format is None:
                # 按优先级尝试不同格式
                for fmt in ["json", "pickle", "txt"]:
                    filepath = domain_dir / f"{filename}.{fmt}"
                    if filepath.exists():
                        format = fmt
                        break
                else:
                    return {
                        "success": False,
                        "data": None,
                        "filepath": None,
                        "error": f"文件不存在: {filename} (尝试了 .json, .pkl, .txt)"
                    }
            else:
                filepath = domain_dir / f"{filename}.{format.lower()}"
            
            if not filepath.exists():
                return {
                    "success": False,
                    "data": None,
                    "filepath": str(filepath),
                    "error": f"文件不存在: {filepath}"
                }
            
            # 根据格式加载
            if format.lower() == "json":
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            
            elif format.lower() == "pickle":
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
            
            elif format.lower() == "txt":
                with open(filepath, "r", encoding="utf-8") as f:
                    data = f.read()
            
            else:
                return {
                    "success": False,
                    "data": None,
                    "filepath": str(filepath),
                    "error": f"不支持的格式: {format}"
                }
            
            return {
                "success": True,
                "data": data,
                "filepath": str(filepath),
                "error": None
            }
        
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "filepath": None,
                "error": str(e)
            }
    
    def get_filepath(
        self,
        domain: str,
        filename: str,
        case_id: Optional[str] = None,
        format: str = "json"
    ) -> Path:
        """
        获取文件路径（不创建文件）
        
        Args:
            domain: 领域名称
            filename: 文件名（不含扩展名）
            case_id: 可选的题目ID
            format: 文件格式
        
        Returns:
            Path: 文件路径
        """
        domain_dir = self.get_domain_dir(domain, case_id)
        filename = str(filename).strip().replace("/", "_").replace("\\", "_")
        return domain_dir / f"{filename}.{format.lower()}"
    
    def list_results(
        self,
        domain: str,
        case_id: Optional[str] = None,
        pattern: Optional[str] = None
    ) -> list:
        """
        列出领域目录下的所有结果文件
        
        Args:
            domain: 领域名称
            case_id: 可选的题目ID
            pattern: 文件名模式（如 "*.json"）
        
        Returns:
            list: 文件路径列表
        """
        domain_dir = self.get_domain_dir(domain, case_id)
        
        if not domain_dir.exists():
            return []
        
        if pattern:
            files = list(domain_dir.glob(pattern))
        else:
            files = list(domain_dir.glob("*"))
        
        # 只返回文件，不包括目录
        return [str(f) for f in files if f.is_file()]
    
    def delete_result(
        self,
        domain: str,
        filename: str,
        case_id: Optional[str] = None,
        format: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        删除结果文件
        
        Args:
            domain: 领域名称
            filename: 文件名（不含扩展名）
            case_id: 可选的题目ID
            format: 文件格式，如果为 None 则尝试所有格式
        
        Returns:
            dict: 删除结果
        """
        try:
            domain_dir = self.get_domain_dir(domain, case_id)
            filename = str(filename).strip().replace("/", "_").replace("\\", "_")
            
            if format:
                filepath = domain_dir / f"{filename}.{format.lower()}"
                if filepath.exists():
                    filepath.unlink()
                    return {"success": True, "deleted": str(filepath)}
                else:
                    return {"success": False, "error": f"文件不存在: {filepath}"}
            else:
                # 尝试所有格式
                deleted = []
                for fmt in ["json", "pickle", "txt", "pkl"]:
                    filepath = domain_dir / f"{filename}.{fmt}"
                    if filepath.exists():
                        filepath.unlink()
                        deleted.append(str(filepath))
                
                if deleted:
                    return {"success": True, "deleted": deleted}
                else:
                    return {"success": False, "error": f"文件不存在: {filename}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}


# 全局单例实例
_global_fs: Optional[EnvironmentFileSystem] = None


def get_environment_fs() -> EnvironmentFileSystem:
    """
    获取全局环境文件系统实例
    
    Returns:
        EnvironmentFileSystem: 全局文件系统管理器实例
    """
    global _global_fs
    if _global_fs is None:
        _global_fs = EnvironmentFileSystem()
    return _global_fs


def save_mid_result(
    domain: str,
    filename: str,
    data: Any,
    case_id: Optional[str] = None,
    format: str = "json",
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：保存中间结果
    
    Args:
        domain: 领域名称
        filename: 文件名
        data: 数据
        case_id: 题目ID
        format: 格式
        **kwargs: 其他参数
    
    Returns:
        dict: 保存结果
    """
    fs = get_environment_fs()
    return fs.save_result(domain, filename, data, case_id, format, **kwargs)


def load_mid_result(
    domain: str,
    filename: str,
    case_id: Optional[str] = None,
    format: Optional[str] = None
) -> Dict[str, Any]:
    """
    便捷函数：加载中间结果
    
    Args:
        domain: 领域名称
        filename: 文件名
        case_id: 题目ID
        format: 格式
    
    Returns:
        dict: 加载结果
    """
    fs = get_environment_fs()
    return fs.load_result(domain, filename, case_id, format)


def get_mid_result_path(
    domain: str,
    filename: str,
    case_id: Optional[str] = None,
    format: str = "json"
) -> Path:
    """
    便捷函数：获取中间结果文件路径
    
    Args:
        domain: 领域名称
        filename: 文件名
        case_id: 题目ID
        format: 格式
    
    Returns:
        Path: 文件路径
    """
    fs = get_environment_fs()
    return fs.get_filepath(domain, filename, case_id, format)
