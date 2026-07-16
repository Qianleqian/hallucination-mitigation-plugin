"""
双模式推理引擎 —— 对应方案文档「阶段2：在线推理双模式分支流程」
整合快速模式（缓存驱动）和精准模式（多源验证），实现动态路由。

快速模式: 本地缓存 -> LLM 生成 -> 免责声明
精准模式: 缓存复用 -> 查询改写 -> 多源搜索 -> 事实验证 -> 缓存更新 -> 约束生成
"""
import json
import time
from typing import AsyncIterator, Optional

from config.settings import config, WorkMode
from core.cache_manager import CacheManager
from core.fact_verifier import FactVerifier, FactClaim
from core.preprocessor import InputPreprocessor, PreprocessResult
from core.search_aggregator import SearchAggregator
from models.llm_base import BaseLLM, LLMResponse
from utils.metrics import ResponseMetrics, metrics_collector
from utils.text_utils import format_uncertainty_disclaimer


class DualModeEngine:
    """
    双模式推理引擎。
    根据用户查询自动选择快速模式或精准模式。

    快速模式（2.1）目标: <200ms 响应，依赖缓存
    精准模式（2.2）目标: 准确率 >90%，依赖多源验证
    """

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.cache = CacheManager()
        self.preprocessor = InputPreprocessor(llm)
        self.searcher = SearchAggregator()
        self.verifier = FactVerifier()
        self._progress_callback = None

    async def ask(
        self,
        query: str,
        mode: Optional[str] = None,
        stream_callback: Optional[callable] = None,
    ) -> dict:
        """
        核心问答入口。
        流程: 预处理 -> 模式决策 -> [快速|精准]推理 -> 记录指标
        """
        start_time = time.time()
        self._progress_callback = stream_callback

        # -- 阶段1：预处理 --
        pre = await self.preprocessor.preprocess(query)
        await self._notify(f"预处理完成 | 意图: {pre.intent} | 不确定性: {pre.uncertainty_score:.2f}")

        # -- 模式决策 --
        if mode is None:
            mode = pre.suggested_mode
        if pre.is_high_risk and mode == "fast":
            await self._notify("检测到高风险领域，强制切换精准模式")
            mode = "precision"

        # 逻辑/常识类 → 专用提示词路径（不走检索），用户手动覆盖除外
        if mode == "logic" and pre.is_logic_question:
            result = await self._logic_mode(pre)
            result["latency_ms"] = (time.time() - start_time) * 1000
            result["mode"] = "logic"
            return result

        effective_mode = WorkMode(mode)

        # -- 阶段2：执行推理 --
        if effective_mode == WorkMode.FAST:
            result = await self._fast_mode(pre)
        else:
            result = await self._precision_mode(pre)

        # -- 记录指标 --
        latency_ms = (time.time() - start_time) * 1000
        metrics_collector.record(ResponseMetrics(
            query=query, mode=effective_mode.value,
            response=result["answer"],
            latency_ms=latency_ms,
            cache_hit=result.get("cache_hit", False),
            sources=result.get("sources", []),
            confidence_score=result.get("confidence", 0.0),
            fact_verification_passed=result.get("verified", False),
        ))
        result["latency_ms"] = latency_ms
        result["mode"] = effective_mode.value
        return result

    async def ask_stream(
        self,
        query: str,
        mode: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """
        流式问答（阶段4：WebSocket 实时进度推送）。
        每个 yield 都是一个进度事件或文本片段。
        """
        pre = await self.preprocessor.preprocess(query)
        effective_mode = WorkMode(mode or pre.suggested_mode)

        if effective_mode == WorkMode.FAST:
            async for chunk in self._fast_mode_stream(pre):
                yield chunk
        else:
            async for chunk in self._precision_mode_stream(pre):
                yield chunk

    # ===================================================
    # 逻辑模式（新增）—— 不走检索，专用提示词直答
    # ===================================================

    async def _logic_mode(self, pre: PreprocessResult) -> dict:
        """
        逻辑/常识类专用路径:
        - 使用逻辑判断专用提示词约束模型行为
        - 不触发多源搜索（更快，且避免引入干扰信息）
        - 不查缓存（逻辑问题的上下文依赖性低）
        """
        await self._notify(
            f"逻辑模式: 专用提示词路径 | 触发: {', '.join(pre.logic_indicators)}"
        )

        system_prompt = pre.logic_prompt or (
            "你是一个逻辑判断专家。请用常识和逻辑直接作答，"
            "一句话给结论，不要展开理论讨论。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pre.cleaned_query},
        ]
        resp = await self.llm.chat(messages, temperature=0.05, max_tokens=400)

        return {
            "answer": resp.content,
            "confidence": 0.85,
            "sources": [],
            "cache_hit": False,
            "verified": False,
            "logic_indicators": pre.logic_indicators,
        }

    async def _logic_mode_stream(self, pre: PreprocessResult):
        yield {"type": "status", "data": f"逻辑判断: {', '.join(pre.logic_indicators)}"}
        result = await self._logic_mode(pre)
        yield {"type": "answer", "data": result["answer"]}
        yield {"type": "done", "data": result}

    # ===================================================
    # 快速模式（2.1）
    # ===================================================

    async def _fast_mode(self, pre: PreprocessResult) -> dict:
        """
        快速模式流程（对应文档 2.1）:
        2.1 查询本地事实缓存
        2.2 判断缓存有效性（阈值 0.7 + 时效）
        2.3 缓存命中 -> 注入系统提示生成
        2.4 缓存未命中 -> 直接调用 LLM
        2.5 附加不确定性免责声明
        """
        await self._notify("快速模式：检查本地缓存...")

        # 2.1-2.2: 查询缓存
        cached = self.cache.search(
            pre.cleaned_query,
            threshold=config.fast_mode_confidence_threshold,
        )

        if cached:
            await self._notify(f"缓存命中！置信度: {cached.confidence:.2f}")
            return {
                "answer": cached.answer + format_uncertainty_disclaimer("medium"),
                "confidence": cached.confidence,
                "sources": cached.source_urls,
                "cache_hit": True,
                "verified": cached.verified,
            }

        # 2.3-2.5: 未命中缓存，直接调用 LLM
        await self._notify("缓存未命中，调用千问生成回答...")

        messages = [
            {"role": "system", "content": "你是一个有帮助的AI助手。请在回答事实性问题时保持谨慎。"},
            {"role": "user", "content": pre.cleaned_query},
        ]
        resp = await self.llm.chat(messages)
        answer = resp.content + format_uncertainty_disclaimer("low")

        return {
            "answer": answer,
            "confidence": 0.3,
            "sources": [],
            "cache_hit": False,
            "verified": False,
        }

    async def _fast_mode_stream(self, pre: PreprocessResult) -> AsyncIterator[dict]:
        """快速模式流式"""
        yield {"type": "status", "data": "快速模式：检查缓存..."}
        result = await self._fast_mode(pre)
        yield {"type": "answer", "data": result["answer"]}
        yield {"type": "done", "data": result}

    # ===================================================
    # 精准模式（2.2）
    # ===================================================

    async def _precision_mode(self, pre: PreprocessResult) -> dict:
        """
        精准模式流程（对应文档 2.2）:
        2.1 查询本地事实缓存（阈值 0.85）
        2.2 高置信缓存 -> 直接复用，跳过后续
        2.3 查询改写优化
        2.4 多源联网搜索（并行，5秒超时）
        2.5 多源交叉验证与去重
        2.6 提取结构化事实
        2.7 更新本地缓存
        2.8 基于验证事实生成最终答案
        """
        await self._notify("精准模式：开始多源验证流程...")

        # 2.1-2.2: 先查缓存（更高阈值）
        cached = self.cache.search(
            pre.cleaned_query,
            threshold=config.precision_mode_confidence_threshold,
        )
        if cached and cached.verified:
            await self._notify(f"精准模式缓存命中（已验证）")
            return {
                "answer": cached.answer + format_uncertainty_disclaimer("high"),
                "confidence": cached.confidence,
                "sources": cached.source_urls,
                "cache_hit": True,
                "verified": True,
            }

        # 2.3: 查询改写
        await self._notify("查询改写优化...")
        rewritten_queries = await self.searcher.rewrite_query(
            pre.cleaned_query, self.llm
        )

        # 2.4: 多源并行搜索
        await self._notify(f"并行搜索 {len(config.search_engines)} 个引擎...")
        all_results = []
        for rq in rewritten_queries:
            search_results = await self.searcher.search(rq)
            all_results.append(search_results)
            await self._notify(f"  '{rq[:40]}...' -> {len(search_results.results)} 条结果")

        # 合并所有搜索结果
        merged_text = "\n\n".join(r.get_text_corpus() for r in all_results)
        all_urls = []
        for r in all_results:
            all_urls.extend(r.get_urls())

        # 2.5: 提取事实声明并进行交叉验证
        await self._notify("提取与交叉验证事实...")
        claims = self.verifier.extract_facts_from_search_results(
            merged_text, source_url=""
        )

        # 先用 LLM 生成原始回答
        raw_answer = await self._generate_constrained_answer(
            pre.cleaned_query, merged_text
        )

        # NLI 验证
        verification = await self.verifier.verify(claims, raw_answer)

        # 2.6-2.7: 基于验证结果调整答案
        if verification.final_fact:
            final_answer = await self._generate_with_fact_injection(
                pre.cleaned_query, verification.final_fact, all_urls
            )
        elif verification.is_verified:
            final_answer = raw_answer + format_uncertainty_disclaimer("high")
        else:
            # 验证未通过：重新生成，加入检索到的所有事实
            final_answer = await self._generate_constrained_answer(
                pre.cleaned_query, merged_text, strict=True
            )
            final_answer += format_uncertainty_disclaimer("medium")

        # 2.8: 更新缓存
        self.cache.store(
            query=pre.cleaned_query,
            answer=final_answer,
            confidence=verification.confidence,
            source_urls=all_urls[:10],
            verified=verification.is_verified,
        )

        await self._notify(f"验证完成 | 置信度: {verification.confidence:.2f} | 通过: {verification.is_verified}")

        return {
            "answer": final_answer,
            "confidence": verification.confidence,
            "sources": all_urls[:10],
            "cache_hit": False,
            "verified": verification.is_verified,
        }

    async def _precision_mode_stream(
        self, pre: PreprocessResult
    ) -> AsyncIterator[dict]:
        """精准模式流式推送"""
        yield {"type": "status", "data": "精准模式：检查缓存..."}

        cached = self.cache.search(
            pre.cleaned_query,
            threshold=config.precision_mode_confidence_threshold,
        )
        if cached and cached.verified:
            yield {"type": "answer", "data": cached.answer}
            yield {"type": "done", "data": {"cache_hit": True}}
            return

        yield {"type": "status", "data": "正在检索多源信息..."}
        rewritten = await self.searcher.rewrite_query(pre.cleaned_query, self.llm)

        for rq in rewritten:
            yield {"type": "status", "data": f"搜索: {rq[:60]}..."}
            results = await self.searcher.search(rq)
            yield {
                "type": "search_result",
                "data": {
                    "query": rq,
                    "count": len(results.results),
                    "urls": results.get_urls()[:5],
                }
            }

        yield {"type": "status", "data": "正在交叉验证事实..."}
        result = await self._precision_mode(pre)
        yield {"type": "answer", "data": result["answer"]}
        yield {"type": "done", "data": result}

    # ===================================================
    # 约束生成
    # ===================================================

    async def _generate_constrained_answer(
        self, query: str, context: str, strict: bool = False
    ) -> str:
        """基于检索到的信息生成答案（2.8 约束注入）"""
        strict_instruction = (
            "你必须严格基于以下参考信息回答问题。"
            "如果参考信息不足以回答，请明确说明'根据现有信息无法确认'。"
            "严禁编造参考信息中不存在的事实。"
            if strict else
            "请参考以下信息回答问题。如果参考信息不足，可以使用你的知识补充。"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    f"你是一个回答事实性问题的AI助手。\n"
                    f"{strict_instruction}\n"
                    f"请在回答末尾标注信息来源。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"参考信息:\n{context[:4000]}\n\n"
                    f"用户问题: {query}\n\n"
                    f"请基于以上信息回答:"
                )
            },
        ]
        resp = await self.llm.chat(messages, temperature=0.05)
        return resp.content

    async def _generate_with_fact_injection(
        self, query: str, verified_fact: str, sources: list
    ) -> str:
        """基于验证后的确定事实生成答案"""
        sources_text = "\n".join(f"- {s}" for s in sources[:5])
        messages = [
            {
                "role": "system",
                "content": (
                    "你必须基于以下已核验的事实来回答。"
                    "不得偏离或编造以下事实之外的内容。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"已核验事实:\n{verified_fact}\n\n"
                    f"信息来源:\n{sources_text}\n\n"
                    f"用户问题: {query}\n\n"
                    f"请基于以上核验事实回答，并标注来源:"
                )
            },
        ]
        resp = await self.llm.chat(messages, temperature=0.05)
        return resp.content + format_uncertainty_disclaimer("high")

    # ===================================================
    # 进度推送
    # ===================================================

    async def _notify(self, message: str):
        """阶段4: 实时进度反馈"""
        if self._progress_callback:
            try:
                await self._progress_callback(message)
            except Exception:
                pass
