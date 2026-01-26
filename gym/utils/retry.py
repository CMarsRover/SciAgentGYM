#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重试装饰器模块

提供通用的重试机制，支持指数退避策略。
"""
import time
from functools import wraps

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # 秒
RETRY_BACKOFF_FACTOR = 2


def retry_on_failure(max_retries=DEFAULT_MAX_RETRIES, delay=DEFAULT_RETRY_DELAY, 
                    backoff_factor=RETRY_BACKOFF_FACTOR, exceptions=(Exception,)):
    """
    重试装饰器，当函数抛出指定异常时进行重试
    
    Args:
        max_retries: 最大重试次数
        delay: 初始重试延迟（秒）
        backoff_factor: 延迟递增因子
        exceptions: 需要重试的异常类型元组
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            raise last_exception
        return wrapper
    return decorator
