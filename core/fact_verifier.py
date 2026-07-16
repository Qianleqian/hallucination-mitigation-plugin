"""
事实验证模块 —— 对应方案文档「阶段2.2：精准模式 - 交叉验证与去重」
使用微型 DeBERTa NLI 模型 + 加权投票机制校验事实一致性。
对冲突信息按「官方 > 学术 > 权威媒体 > 普通资讯」优先级排序。
"""
import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config.settings import config


@dataclass
class FactClaim:
    """从文本中提取的事实声明"""
    text: str
    source_url: str
    source_type: str = "普通资讯"  # 官方/学术/权威媒体/普通资讯
    supporting_evidence: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """验证结果"""
    claim: str
    is_verified: bool
    confidence: float
    supporting_sources: List[str]
    conflicting_sources: List[str]
    final_fact: Optional[str] = None
    source_priority_used: bool = False


class FactVerifier:
    """
    事实验证器。
    使用 NLI 模型进行事实蕴含检测，多源加权投票。

    流程: 提取事实 -> NLI 校对 -> 冲突排序 -> 合成最终事实
    """

    def __init__(self):
        self._nli_model = None
        self._nli_tokenizer = None
        self._source_priority = {
            "官方": 4, "学术": 3, "权威媒体": 2, "普通资讯": 1,
        }

    @property
    def nli_model(self):
        if self._nli_model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                model_name = config.nli_model_name
                self._nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                print(f"[验证] NLI 模型已加载: {model_name}")
            except Exception as e:
                print(f"[验证] NLI 模型加载失败，使用启发式退避: {e}")
                self._nli_model = False  # 标记为不可用
        return self._nli_model

    async def verify(
        self,
        claims: List[FactClaim],
        original_answer: str,
    ) -> VerificationResult:
        """
        对 LLM 生成的回答进行事实验证。

        Args:
            claims: 从多源搜索结果中提取的事实声明
            original_answer: LLM 原始回答

        Returns:
            VerificationResult 包含验证结果和最终建议
        """
        if not claims:
            return VerificationResult(
                claim=original_answer[:200],
                is_verified=False,
                confidence=0.0,
                supporting_sources=[],
                conflicting_sources=[],
            )

        # 对每个 claim 进行 NLI 验证
        verified_claims = []
        conflicting_claims = []

        for claim in claims:
            nli_score = await self._nli_check(original_answer, claim.text)
            if nli_score >= config.fact_verification_threshold:
                verified_claims.append((claim, nli_score))
            else:
                conflicting_claims.append((claim, nli_score))

        # 加权投票
        supporting_sources = [c.source_url for c, _ in verified_claims]
        conflicting_sources = [c.source_url for c, _ in conflicting_claims]

        # 对冲突信息按来源优先级排序
        if conflicting_claims:
            conflicting_claims.sort(
                key=lambda x: self._source_priority.get(x[0].source_type, 1),
                reverse=True,
            )

        # 确定最终事实
        is_verified = len(verified_claims) >= len(conflicting_claims)
        confidence = (
            sum(s for _, s in verified_claims) / len(verified_claims)
            if verified_claims else 0.0
        )

        final_fact = None
        if conflicting_claims and self._source_priority.get(
            conflicting_claims[0][0].source_type, 1
        ) >= 3:
            # 高优先级来源的冲突信息覆盖原回答
            final_fact = conflicting_claims[0][0].text
            is_verified = True

        return VerificationResult(
            claim=original_answer[:200],
            is_verified=is_verified,
            confidence=min(confidence, 1.0),
            supporting_sources=supporting_sources,
            conflicting_sources=conflicting_sources,
            final_fact=final_fact,
            source_priority_used=bool(conflicting_claims),
        )

    async def _nli_check(self, premise: str, hypothesis: str) -> float:
        """
        NLI 蕴含检测。
        premise = LLM 回答（前提）
        hypothesis = 搜索结果片段（假设）
        返回 entailment 得分 (0-1)，越高表示假设支持前提。
        """
        if self.nli_model is None or self.nli_model is False:
            return self._heuristic_check(premise, hypothesis)

        try:
            import torch

            inputs = self._nli_tokenizer(
                premise[:512], hypothesis[:512],
                truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.nli_model(**inputs).logits
                # DeBERTa NLI: [contradiction, neutral, entailment]
                probs = torch.softmax(logits, dim=-1)[0]
                entailment_score = float(probs[2])  # entailment 概率

            return entailment_score
        except Exception:
            return self._heuristic_check(premise, hypothesis)

    def _heuristic_check(self, text1: str, text2: str) -> float:
        """
        启发式文本相似度（NLI 模型不可用时的降级方案）
        基于关键词重叠 + 简单语义启发。
        """
        from utils.text_utils import extract_keywords

        kw1 = set(extract_keywords(text1, top_n=15))
        kw2 = set(extract_keywords(text2, top_n=15))

        if not kw1 or not kw2:
            return 0.5

        overlap = len(kw1 & kw2)
        union = len(kw1 | kw2)
        jaccard = overlap / union if union > 0 else 0

        # 额外检查数字匹配（数字事实尤其重要）
        nums1 = set(re.findall(r"\d+", text1))
        nums2 = set(re.findall(r"\d+", text2))
        if nums1 and nums2:
            num_overlap = len(nums1 & nums2) / max(len(nums1), len(nums2))
            jaccard = 0.7 * jaccard + 0.3 * num_overlap

        return jaccard

    @staticmethod
    def extract_facts_from_search_results(
        search_text: str, source_url: str = ""
    ) -> List[FactClaim]:
        """从搜索结果文本中提取事实声明"""
        sentences = re.split(r"[。.！!？?\n]+", search_text)
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 15:  # 过滤过短片段
                claims.append(FactClaim(
                    text=sent,
                    source_url=source_url,
                    source_type=FactVerifier._guess_source_type(source_url, sent),
                ))
        return claims

    @staticmethod
    def _guess_source_type(url: str, text: str) -> str:
        """根据 URL/文本推测来源类型"""
        url_lower = (url or "").lower()
        if any(d in url_lower for d in [".gov", ".edu", "official"]):
            return "官方"
        if any(d in url_lower for d in ["scholar", "arxiv", "research", "academic", "cnki"]):
            return "学术"
        if any(d in url_lower for d in ["news", "xinhua", "people", "reuters", "bbc", "wsj"]):
            return "权威媒体"
        return "普通资讯"
