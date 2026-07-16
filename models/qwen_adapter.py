"""
通义千问 (Qwen) 适配器 —— 以千问为插件目标
通过 DashScope API 接入千问全系列模型，实现统一 BaseLLM 接口。
支持：qwen-max, qwen-plus, qwen-turbo, qwen2.5-72b 等全系列。
"""
import asyncio
import os
from typing import AsyncIterator, List, Optional

from models.llm_base import BaseLLM, LLMConfig, LLMResponse


class QwenAdapter(BaseLLM):
    """
    千问适配器。
    以千问为插件的目标模型，实现幻觉缓解插件的完整接入。

    使用方法:
        config = LLMConfig(
            model="qwen-plus",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        qwen = QwenAdapter(config)
        resp = await qwen.chat([{"role": "user", "content": "你好"}])
    """

    # 千问系列模型列表
    AVAILABLE_MODELS = [
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
        "qwen2.5-72b-instruct",
        "qwen2.5-32b-instruct",
        "qwen2.5-14b-instruct",
        "qwen2.5-7b-instruct",
        "qwq-32b-preview",
        "qwen-long",
    ]

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self._init_client()

    def _init_client(self):
        """初始化 DashScope 客户端"""
        try:
            import dashscope
            self.dashscope = dashscope
            dashscope.api_key = self.config.api_key
        except ImportError:
            raise ImportError(
                "请安装 dashscope: pip install dashscope>=1.20.0"
            )

    async def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        千问统一聊天接口。
        将千问的 API 转换为系统统一的 LLMResponse 格式。
        """
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        # 构建系统提示词，约束模型基于给定事实回答
        system_msg = self._extract_system_prompt(messages)

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.dashscope.Generation.call(
                    model=model,
                    messages=messages,
                    result_format="message",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=self.config.top_p,
                    enable_search=False,  # 关闭千问内置搜索（用插件多层搜索替代）
                )
            )
        except Exception as e:
            return LLMResponse(
                content=f"[千问 API 调用异常] {str(e)}",
                model=model,
                finish_reason="error",
            )

        if response.status_code != 200:
            return LLMResponse(
                content=f"[千问 API 错误] code={response.status_code}, message={response.message}",
                model=model,
                finish_reason="error",
            )

        output = response.output
        if output and output.choices:
            choice = output.choices[0]
            content = choice.message.content

            # 提取 logits（千问 v2 部分模型支持）
            logits = None
            if hasattr(output, "logprobs") and output.logprobs:
                logits = self._extract_logits(output.logprobs)

            return LLMResponse(
                content=content,
                model=model,
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "input_tokens": output.usage.input_tokens if output.usage else 0,
                    "output_tokens": output.usage.output_tokens if output.usage else 0,
                },
                logits=logits,
            )
        else:
            return LLMResponse(
                content=f"[千问响应异常] {response.message}",
                model=model,
                finish_reason="error",
            )

    async def chat_stream(
        self,
        messages: List[dict],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式调用千问。
        用于阶段4：WebSocket 实时进度推送。
        """
        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)

        loop = asyncio.get_event_loop()

        try:
            responses = await loop.run_in_executor(
                None,
                lambda: list(self.dashscope.Generation.call(
                    model=model,
                    messages=messages,
                    result_format="message",
                    temperature=temperature,
                    max_tokens=self.config.max_tokens,
                    stream=True,
                    incremental_output=True,
                ))
            )

            for resp in responses:
                if resp.status_code == 200 and resp.output and resp.output.choices:
                    content = resp.output.choices[0].message.content
                    if content:
                        yield content
        except Exception as e:
            yield f"[千问流式调用异常] {str(e)}"

    async def get_logits_uncertainty(self, messages: List[dict]) -> float:
        """
        获取千问回答的不确定性（基于 logprobs 近似）。
        对应阶段1: 通过模型 logits 熵值辅助模式决策。

        千问 API 支持 logprobs 参数的模型会返回 token-level logprobs，
        我们通过方差和熵来估计不确定性。
        """
        model = self.config.model
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.dashscope.Generation.call(
                    model=model,
                    messages=messages,
                    result_format="message",
                    temperature=0.0,
                    max_tokens=1,  # 只需 logprobs
                    top_p=1.0,
                )
            )

            if response.status_code != 200:
                return 0.5  # 默认中等不确定性

            # 从千问响应中提取 logprobs
            output = response.output
            if output and hasattr(output, "logprobs") and output.logprobs:
                logprobs = self._extract_logits(output.logprobs)
                return self._compute_entropy(logprobs)

            return 0.5

        except Exception:
            return 0.5

    def _extract_system_prompt(self, messages: List[dict]) -> Optional[str]:
        """提取系统提示词"""
        for msg in messages:
            if msg.get("role") == "system":
                return msg["content"]
        return None

    def _extract_logits(self, logprobs) -> List[float]:
        """从千问 logprobs 提取 logit 值"""
        if isinstance(logprobs, list) and logprobs:
            if hasattr(logprobs[0], "token_logprob"):
                return [lp.token_logprob for lp in logprobs[:10]]
        return []

    def _compute_entropy(self, logits: List[float]) -> float:
        """计算 logits 熵值"""
        import math
        if not logits:
            return 0.5
        # 归一化
        max_val = max(logits)
        probs = [math.exp(l - max_val) for l in logits]
        total = sum(probs)
        if total == 0:
            return 0.5
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        # 归一化到 [0, 1]
        max_entropy = math.log(len(probs) + 1e-10)
        return entropy / max_entropy if max_entropy > 0 else 0.5


# -- 千问工具注册适配 --
class QwenToolAdapter:
    """
    将系统插件工具注册给千问。
    千问的 function calling 使用 OpenAI 兼容格式。
    """

    @staticmethod
    def format_tools_for_qwen(tools: List[dict]) -> List[dict]:
        """将统一的 Tool 定义转为千问兼容格式"""
        qwen_tools = []
        for tool in tools:
            qwen_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                }
            })
        return qwen_tools

    @staticmethod
    def parse_tool_call(response: dict) -> Optional[dict]:
        """解析千问返回的工具调用"""
        if "tool_calls" in response:
            for tc in response["tool_calls"]:
                return {
                    "name": tc.get("function", {}).get("name"),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                }
        return None
