"""
ToolRegistry —— 对应方案文档「阶段0：插件接入」
基于 MCP 协议理念的工具注册中心，实现插件热插拔与能力发现。

所有外部能力（搜索、验证、缓存）均通过 Tool 形式注册，
LLM 可通过 function calling 自动发现并调用。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable
    category: str = "general"  # search / verification / cache / utility
    requires_auth: bool = False
    timeout_sec: int = 30

    def to_openai_schema(self) -> dict:
        """转为 OpenAI/千问 兼容的 function calling schema"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """
    工具注册中心。
    支持 MCP 协议：工具注册 -> 工具发现 -> 工具调用。

    使用:
        registry = ToolRegistry()
        registry.register(ToolDefinition(...))
        tools = registry.list_tools()
        result = await registry.call("search_web", {"query": "..."})
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._call_history: List[dict] = []

    def register(self, tool: ToolDefinition):
        """注册一个工具"""
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        """移除一个工具"""
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolDefinition]:
        """按名称获取工具"""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[ToolDefinition]:
        """列出所有工具，可按类别筛选"""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def list_openai_schemas(self, category: Optional[str] = None) -> List[dict]:
        """列出所有工具的 OpenAI/千问 schema（用于 function calling）"""
        return [t.to_openai_schema() for t in self.list_tools(category)]

    async def call(self, name: str, arguments: dict) -> Any:
        """调用一个工具"""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"工具 '{name}' 未注册。可用: {list(self._tools.keys())}")

        try:
            import asyncio
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool.handler(**arguments))
        except Exception as e:
            result = {"error": str(e), "tool": name}

        self._call_history.append({
            "tool": name,
            "arguments": arguments,
            "result_summary": str(result)[:200],
        })
        return result

    def get_call_history(self) -> List[dict]:
        return self._call_history


# 全局单例
tool_registry = ToolRegistry()
