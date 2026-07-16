"""
LLM 基类接口 —— 对应方案文档「阶段0：插件接入」
基于 MCP 协议理念设计统一调用接口，实现模型无关化接入。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional


@dataclass
class LLMResponse:
    """统一的 LLM 响应结构"""
    content: str
    model: str
    finish_reason: str = "stop"
    usage: dict = None
    logits: Optional[List[float]] = None  # 用于不确定性检测

    def __post_init__(self):
        if self.usage is None:
            self.usage = {}


@dataclass
class LLMConfig:
    """统一的 LLM 配置"""
    model: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.1  # 低温度以降低幻觉
    top_p: float = 0.9
    extra: dict = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}


class BaseLLM(ABC):
    """
    所有 LLM 适配器的抽象基类。
    遵循 MCP 协议理念：统一调用接口 + 数据交互规范，
    无需针对不同模型单独适配开发。
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._tool_registry = None

    @property
    def tool_registry(self):
        if self._tool_registry is None:
            from models.tool_registry import ToolRegistry
            self._tool_registry = ToolRegistry()
        return self._tool_registry

    @abstractmethod
    async def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """统一聊天接口"""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: List[dict],
        **kwargs
    ) -> AsyncIterator[str]:
        """统一流式接口"""
        ...

    @abstractmethod
    async def get_logits_uncertainty(self, messages: List[dict]) -> float:
        """获取模型对回答的 logits 熵值/不确定性（阶段1预处理）"""
        ...
