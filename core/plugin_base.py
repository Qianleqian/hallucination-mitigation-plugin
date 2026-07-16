"""
插件基类 —— 对应方案文档「阶段0：插件接入」
基于 MCP 协议 + ToolRegistry 架构实现模型无关化接入。
"""
from abc import ABC, abstractmethod
from typing import Optional

from models.llm_base import BaseLLM
from models.tool_registry import ToolRegistry


class HallucinationMitigationPlugin(ABC):
    """
    幻觉缓解插件基类。
    所有 LLM 的幻觉缓解插件均继承此类，实现统一接口。

    使用:
        plugin = QwenHallucinationPlugin(llm=qwen_adapter, config=config)
        response = await plugin.ask("感冒了怎么办？")
    """

    def __init__(self, llm: BaseLLM, config=None):
        from config.settings import PluginConfig
        self.llm = llm
        self.config = config or PluginConfig()
        self.tool_registry = ToolRegistry()
        self._setup_tools()

    @abstractmethod
    def _setup_tools(self):
        """注册插件使用的工具（搜索、验证、缓存等）"""
        ...

    @abstractmethod
    async def ask(
        self,
        query: str,
        mode: Optional[str] = None,
        stream_callback: Optional[callable] = None,
    ) -> dict:
        """
        核心问答接口。

        Args:
            query: 用户输入
            mode: 工作模式 ("fast" / "precision")，None 则自动决策
            stream_callback: 阶段4 实时进度回调

        Returns:
            {
                "answer": str,
                "mode": str,
                "confidence": float,
                "sources": list,
                "latency_ms": float,
                "cache_hit": bool,
            }
        """
        ...

    @abstractmethod
    async def ask_stream(self, query: str, **kwargs):
        """流式问答（阶段4：WebSocket 实时推送）"""
        ...
