#!/usr/bin/env python3
"""
服务器端测试: 本地千问 + 幻觉缓解插件
纯 Python 实现，无 numpy/transformers 依赖
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
# 简易缓存管理器 (无 numpy 依赖)
# ============================================================

@dataclass
class CacheEntry:
    id: str
    query: str
    answer: str
    confidence: float = 0.8
    source_urls: List[str] = field(default_factory=list)
    verified: bool = False
    created_at: float = field(default_factory=time.time)
    ttl_days: int = 30

    def is_expired(self):
        return (time.time() - self.created_at) > self.ttl_days * 86400

    def is_valid(self, threshold=0.7):
        return self.confidence >= threshold and not self.is_expired()


class SimpleCache:
    def __init__(self):
        self._cache: List[CacheEntry] = []

    def _keywords(self, text: str) -> set:
        stop = {
            "的","了","在","是","我","有","和","就","不","人","都","一",
            "一个","上","也","很","到","说","要","去","你","会","着",
            "the","a","an","is","are","was","were","be","been","have",
            "没有","看","好","自己","这","他","她","它","们","那","些",
        }
        words = re.findall(r"[一-鿿]+|[a-zA-Z]+", text.lower())
        freq = {}
        for w in words:
            if w not in stop and len(w) > 1:
                freq[w] = freq.get(w, 0) + 1
        return set(sorted(freq, key=lambda x: -freq[x])[:15])

    def search(self, query: str, threshold=0.7) -> Optional[CacheEntry]:
        qk = self._keywords(query)
        if not qk:
            return None
        best, best_score = None, 0
        for e in self._cache:
            if not e.is_valid(threshold):
                continue
            ek = self._keywords(e.query)
            if not ek:
                continue
            score = len(qk & ek) / min(len(qk), len(ek))
            if score > best_score:
                best_score = score
                best = e
        return best if best and best_score >= threshold else None

    def store(self, query, answer, confidence=0.8, source_urls=None, verified=False):
        self._cache.append(CacheEntry(
            id=f"c{len(self._cache)}",
            query=query, answer=answer,
            confidence=confidence,
            source_urls=source_urls or [],
            verified=verified,
        ))

    def get_high_quality_entries(self, threshold=0.9):
        return [e for e in self._cache if e.confidence >= threshold and e.verified and not e.is_expired()]

    def evict_expired(self):
        before = len(self._cache)
        self._cache = [e for e in self._cache if not e.is_expired()]
        return before - len(self._cache)

    @property
    def stats(self):
        return {"total_entries": len(self._cache)}


# ============================================================
# 本地千问客户端 (vLLM OpenAI 兼容 API)
# ============================================================

class QwenClient:
    def __init__(self, base_url="http://127.0.0.1:8009/v1", model="qwen35-9b-thinking"):
        self.base_url = base_url
        self.model = model

    async def chat(self, messages: List[dict], temperature=0.1, max_tokens=1000) -> str:
        import urllib.request
        loop = asyncio.get_event_loop()
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        def _req():
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))

        data = await loop.run_in_executor(None, _req)
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content")
        if content:
            return content.strip()
        reasoning = msg.get("reasoning", "")
        return reasoning[-500:].strip() if reasoning else "(模型未返回内容)"

    def get_model_info(self) -> dict:
        import urllib.request
        req = urllib.request.Request(f"{self.base_url}/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ============================================================
# 幻觉缓解引擎
# ============================================================

class HallucinationMitigationEngine:
    def __init__(self, llm: QwenClient):
        self.llm = llm
        self.cache = SimpleCache()

    async def ask(self, query: str, mode: str = "fast") -> dict:
        start = time.time()

        if mode == "fast":
            # 快速模式: 缓存优先
            cached = self.cache.search(query, threshold=0.6)
            if cached:
                return {
                    "answer": cached.answer,
                    "mode": "fast",
                    "latency_ms": (time.time() - start) * 1000,
                    "cache_hit": True,
                    "verified": cached.verified,
                    "confidence": cached.confidence,
                }

            answer = await self.llm.chat([
                {"role": "system", "content": "你是一个有帮助的AI助手，请简洁回答。"},
                {"role": "user", "content": query},
            ], max_tokens=500)
            return {
                "answer": answer,
                "mode": "fast",
                "latency_ms": (time.time() - start) * 1000,
                "cache_hit": False,
                "verified": False,
                "confidence": 0.3,
            }

        else:
            # 精准模式: 约束生成 + 自检验证
            system_prompt = (
                "你是一个严格基于事实的AI助手。回答问题时:\n"
                "1. 只陈述你确定的事实\n"
                "2. 如果不确定，明确说'我无法确定'\n"
                "3. 区分事实和观点\n"
                "4. 如果有数据，给出具体数字和来源"
            )

            answer = await self.llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ], max_tokens=800)

            # 自检验证
            verify_prompt = (
                f"请检查以下回答是否存在事实性错误。"
                f"如果内容正确请回复'正确'，如果有不确定的地方请指出。\n\n"
                f"用户问题: {query}\n"
                f"AI回答: {answer}\n\n请评估:"
            )
            check = await self.llm.chat([
                {"role": "user", "content": verify_prompt},
            ], max_tokens=300, temperature=0.0)

            is_verified = "正确" in check and "错误" not in check

            # 缓存
            self.cache.store(
                query=query, answer=answer,
                confidence=0.85 if is_verified else 0.5,
                source_urls=["local_qwen_vllm"],
                verified=is_verified,
            )

            return {
                "answer": answer,
                "mode": "precision",
                "latency_ms": (time.time() - start) * 1000,
                "cache_hit": False,
                "verified": is_verified,
                "confidence": 0.85 if is_verified else 0.5,
                "self_check": check[:300],
            }


# ============================================================
# 主测试
# ============================================================

async def main():
    print("=" * 60)
    print("  大模型幻觉缓解插件 - 服务器端测试")
    print("  模型: Qwen3.5-9B-Thinking (vLLM @ 127.0.0.1:8009)")
    print("=" * 60)

    # 连接模型
    print("\n[初始化] 连接本地千问服务...")
    qwen = QwenClient()
    model_info = qwen.get_model_info()
    model_name = model_info.get("data", [{}])[0].get("id", "unknown")
    print(f"  模型: {model_name}")
    print("  [OK] 千问服务连接成功")

    engine = HallucinationMitigationEngine(qwen)

    # 测试用例
    tests = [
        ("fast", "请用一句话介绍你自己"),
        ("fast", "什么是大模型幻觉？请简要说明"),
        ("precision", "请解释Python编程语言的核心特点"),
        ("precision", "标准大气压下水的沸点是多少度？"),
        ("precision", "地球到月球的距离大约是多少？"),
    ]

    for mode, query in tests:
        print(f"\n{'─' * 55}")
        print(f"  [{mode.upper()}] 问题: {query}")
        result = await engine.ask(query, mode=mode)
        print(f"  延迟: {result['latency_ms']:.0f}ms")
        print(f"  缓存命中: {result['cache_hit']}")
        if not result['cache_hit']:
            print(f"  事实验证: {result.get('verified', False)}")
            print(f"  置信度: {result.get('confidence', 0):.2f}")
        answer_preview = result['answer'][:400].replace('\n', ' ')
        print(f"  回答: {answer_preview}...")

    # 第二次同样问题 - 验证缓存
    print(f"\n{'─' * 55}")
    print("  [缓存验证] 重复提问: '什么是大模型幻觉？'")
    result = await engine.ask("什么是大模型幻觉？请简要说明", mode="fast")
    print(f"  缓存命中: {result['cache_hit']}")
    print(f"  延迟: {result['latency_ms']:.0f}ms")

    print(f"\n{'=' * 60}")
    print(f"  缓存统计: {engine.cache.stats}")
    print("  [OK] 全部测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
