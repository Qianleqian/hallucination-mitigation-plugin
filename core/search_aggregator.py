"""
多源搜索聚合器 —— 对应方案文档「阶段2.2：精准模式 - 多源联网检索」
并行调用 2-3 个搜索引擎，5 秒超时控制，返回去重后的搜索结果。
支持：DuckDuckGo（免费）、Bing API、SerpAPI（可选）。
"""
import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from config.settings import config


@dataclass
class SearchResult:
    """单条搜索结果"""
    title: str
    url: str
    snippet: str
    source_engine: str
    rank: int = 0
    timestamp: Optional[str] = None

    def dedup_key(self) -> str:
        return hashlib.md5(
            (self.url or self.title).encode("utf-8")
        ).hexdigest()


@dataclass
class AggregatedSearchResults:
    """聚合后的搜索结果"""
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_sources: int = 0
    elapsed_ms: float = 0.0

    def get_text_corpus(self) -> str:
        """合并所有搜索片段为一段文本"""
        texts = []
        for r in self.results:
            texts.append(f"[来源: {r.source_engine}] {r.title}\n{r.snippet}")
        return "\n\n".join(texts)

    def get_urls(self) -> List[str]:
        return list({r.url for r in self.results if r.url})


class SearchAggregator:
    """
    多源搜索聚合器。
    并行调用搜索引擎，超时控制，结果去重排序。
    """

    def __init__(self):
        self._engines = {
            "duckduckgo": self._search_duckduckgo,
            "bing": self._search_bing,
            "serpapi": self._search_serpapi,
        }

    async def search(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        max_results: int = None,
    ) -> AggregatedSearchResults:
        """
        并行调用多个搜索引擎。
        最多 5 秒超时，保证用户体验。
        """
        import time
        start = time.time()

        engines_to_use = engines or config.search_engines
        max_per_engine = max_results or config.max_search_results_per_engine
        timeout = config.search_timeout_sec

        tasks = []
        for engine_name in engines_to_use:
            if engine_name in self._engines:
                tasks.append(self._search_with_timeout(
                    engine_name, query, max_per_engine, timeout
                ))

        if not tasks:
            return AggregatedSearchResults(query=query)

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并去重
        seen = set()
        all_results = []
        for result_list in results_lists:
            if isinstance(result_list, Exception):
                print(f"[搜索] 引擎异常: {result_list}")
                continue
            for r in (result_list or []):
                key = r.dedup_key()
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)

        # 按排名排序
        all_results.sort(key=lambda r: r.rank)

        elapsed = (time.time() - start) * 1000

        return AggregatedSearchResults(
            query=query,
            results=all_results[:max_per_engine * len(engines_to_use)],
            total_sources=len(engines_to_use),
            elapsed_ms=elapsed,
        )

    async def _search_with_timeout(
        self, engine_name: str, query: str, max_results: int, timeout: int
    ) -> List[SearchResult]:
        """带超时的单引擎搜索"""
        try:
            return await asyncio.wait_for(
                self._search_engine(engine_name, query, max_results),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            print(f"[搜索] {engine_name} 超时 ({timeout}秒)")
            return []
        except Exception as e:
            print(f"[搜索] {engine_name} 异常: {e}")
            return []

    async def _search_engine(
        self, engine_name: str, query: str, max_results: int
    ) -> List[SearchResult]:
        """单引擎搜索调度"""
        handler = self._engines.get(engine_name)
        if handler is None:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, handler, query, max_results)

    def _search_duckduckgo(
        self, query: str, max_results: int
    ) -> List[SearchResult]:
        """DuckDuckGo 搜索（免费，无需 API Key）"""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("[搜索] 请安装 duckduckgo-search: pip install duckduckgo-search")
            return []

        results = []
        try:
            with DDGS() as ddgs:
                for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source_engine="duckduckgo",
                        rank=i,
                    ))
        except Exception as e:
            print(f"[搜索] DuckDuckGo 请求失败: {e}")

        return results

    def _search_bing(self, query: str, max_results: int) -> List[SearchResult]:
        """Bing Web Search API（需要 API Key）"""
        import os
        api_key = os.getenv("BING_API_KEY")
        if not api_key:
            return []  # 静默跳过

        try:
            import httpx
            resp = httpx.get(
                "https://api.bing.microsoft.com/v7.0/search",
                params={"q": query, "count": max_results, "mkt": "zh-CN"},
                headers={"Ocp-Apim-Subscription-Key": api_key},
                timeout=5,
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = []
            for i, item in enumerate(data.get("webPages", {}).get("value", [])):
                results.append(SearchResult(
                    title=item.get("name", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source_engine="bing",
                    rank=i,
                ))
            return results
        except Exception as e:
            print(f"[搜索] Bing API 异常: {e}")
            return []

    def _search_serpapi(self, query: str, max_results: int) -> List[SearchResult]:
        """SerpAPI 搜索（需要 API Key）"""
        import os
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return []

        try:
            import httpx
            resp = httpx.get(
                "https://serpapi.com/search",
                params={
                    "q": query, "api_key": api_key,
                    "num": max_results, "hl": "zh-CN",
                },
                timeout=5,
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            results = []
            for i, item in enumerate(
                data.get("organic_results", [])[:max_results]
            ):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source_engine="serpapi",
                    rank=i,
                ))
            return results
        except Exception as e:
            print(f"[搜索] SerpAPI 异常: {e}")
            return []

    async def rewrite_query(self, query: str, llm=None) -> List[str]:
        """
        查询改写优化（阶段2.2 精准模式子步骤 2.3）。
        通过 LLM 将用户查询扩展为多维度检索指令，
        扩大信息召回范围。
        """
        if llm is None:
            return [query]

        prompt = f"""请将以下用户查询改写为 2-3 个不同角度的搜索查询，
用于多源信息检索。只返回搜索查询，每行一个，不要编号。

用户查询: {query}

搜索查询:"""

        try:
            resp = await llm.chat([
                {"role": "user", "content": prompt}
            ])
            lines = [l.strip("- 123456789. ") for l in resp.content.strip().split("\n")
                     if l.strip() and l.strip() != "搜索查询:"]
            return lines[:3] or [query]
        except Exception:
            return [query]
