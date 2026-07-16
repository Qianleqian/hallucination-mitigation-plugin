"""
本地千问适配器 —— 连接 vLLM 部署的千问模型
通过 OpenAI 兼容 API 接入本地千问，无需 DashScope Key。

服务器环境: vLLM + qwen35-9b-thinking @ 127.0.0.1:8009
"""
import asyncio
import json
import os
from typing import AsyncIterator, List, Optional

from models.llm_base import BaseLLM, LLMConfig, LLMResponse


class LocalQwenAdapter(BaseLLM):
    """
    本地千问适配器。
    通过 vLLM 的 OpenAI 兼容 API 连接本地千问模型。

    使用方法:
        config = LLMConfig(
            model="qwen35-9b-thinking",
            api_key="not-needed",  # 本地部署无需 Key
            base_url="http://127.0.0.1:8009/v1",
        )
        qwen = LocalQwenAdapter(config)
        resp = await qwen.chat([{"role": "user", "content": "你好"}])
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self.base_url = config.base_url or "http://127.0.0.1:8009/v1"

    async def chat(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        **kwargs
    ) -> LLMResponse:
        """通过 HTTP 调用本地 vLLM API"""
        import urllib.request
        import urllib.error

        model = kwargs.get("model", self.config.model)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": self.config.top_p,
            # 关闭 thinking 模式以加快响应速度，减少 token 消耗
            "chat_template_kwargs": {"enable_thinking": False},
        }

        if tools:
            payload["tools"] = tools

        loop = asyncio.get_event_loop()

        try:
            response_data = await loop.run_in_executor(
                None,
                lambda: self._make_request(payload)
            )

            if "error" in response_data:
                return LLMResponse(
                    content=f"[本地千问错误] {response_data['error']}",
                    model=model,
                    finish_reason="error",
                )

            choice = response_data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content") or ""

            # 如果 content 为空但有 reasoning，使用 reasoning 作为回答
            if not content and message.get("reasoning"):
                content = message["reasoning"][-500:]  # 取最后部分作为回答

            usage = response_data.get("usage", {})

            return LLMResponse(
                content=content.strip() if content else "(模型未返回内容)",
                model=model,
                finish_reason=choice.get("finish_reason", "stop"),
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
            )

        except Exception as e:
            return LLMResponse(
                content=f"[本地千问调用异常] {str(e)}",
                model=model,
                finish_reason="error",
            )

    def _make_request(self, payload: dict) -> dict:
        """同步 HTTP 请求"""
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}

    async def chat_stream(
        self,
        messages: List[dict],
        **kwargs
    ) -> AsyncIterator[str]:
        """流式调用（vLLM 支持 SSE）"""
        model = kwargs.get("model", self.config.model)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        loop = asyncio.get_event_loop()
        url = f"{self.base_url}/chat/completions"

        import urllib.request
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
        )

        try:
            resp = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(req, timeout=120)
            )
            for line in resp:
                line = line.decode("utf-8").strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"[流式异常] {str(e)}"

    async def get_logits_uncertainty(self, messages: List[dict]) -> float:
        """通过 logprobs 获取不确定性（vLLM 支持）"""
        try:
            loop = asyncio.get_event_loop()
            payload = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": True,
                "top_logprobs": 10,
            }
            response_data = await loop.run_in_executor(
                None, lambda: self._make_request(payload)
            )

            if "error" in response_data:
                return 0.5

            logprobs_data = response_data.get("choices", [{}])[0].get("logprobs", {})
            if logprobs_data and "content" in logprobs_data:
                top_logprobs = logprobs_data["content"][0].get("top_logprobs", [])
                if top_logprobs:
                    import math
                    probs = [math.exp(lp.get("logprob", 0)) for lp in top_logprobs]
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                        max_entropy = math.log(len(probs) + 1e-10)
                        return entropy / max_entropy if max_entropy > 0 else 0.5

            return 0.5
        except Exception:
            return 0.5
