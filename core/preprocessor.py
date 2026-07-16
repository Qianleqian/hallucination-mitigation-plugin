"""
输入预处理器 —— 对应方案文档「阶段1：用户输入与预处理」
完成数据清洗、意图识别、不确定性预判，为后续模式分支决策提供依据。
"""
from dataclasses import dataclass
from typing import Optional

from models.llm_base import BaseLLM
from utils.text_utils import clean_text, extract_keywords


@dataclass
class PreprocessResult:
    """预处理结果"""
    cleaned_query: str
    keywords: list
    intent: str                    # 问题意图类型
    uncertainty_score: float       # 0-1，越高越不确定
    suggested_mode: str            # "fast" / "precision"
    is_high_risk: bool
    query_hash: str                # 用于缓存查找


class InputPreprocessor:
    """
    阶段1 预处理器。
    轻量化设计：不添加冗余检测，仅做必要的清洗、意图识别和不确定性评估。
    """

    # 意图分类规则
    INTENT_PATTERNS = {
        "factual":    ["是什么", "什么是", "谁", "何时", "哪里", "多少", "怎么定义",
                       "what is", "who is", "when", "where", "define"],
        "how_to":     ["怎么做", "如何", "怎样", "方法", "步骤", "教程",
                       "how to", "how do", "steps", "guide"],
        "opinion":    ["评价", "怎么样", "好不好", "推荐", "建议",
                       "review", "opinion", "recommend", "best"],
        "comparison": ["对比", "区别", "哪个好", "比较", "vs",
                       "compare", "difference", "versus"],
        "calculation": ["计算", "公式", "等于", "换算",
                        "calculate", "formula", "convert"],
    }

    # 高风险领域关键词（对应 config.HIGH_RISK_DOMAINS 的补充）
    HIGH_RISK_KEYWORDS = [
        "金融", "医疗", "法律", "药品", "手术", "投资",
        "证券", "保险", "税务", "法规", "剂量", "禁忌",
    ]

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm

    async def preprocess(self, raw_query: str) -> PreprocessResult:
        """入口：完成全部预处理"""
        cleaned = clean_text(raw_query)
        keywords = extract_keywords(cleaned)
        intent = self._detect_intent(cleaned)
        is_high_risk = self._check_high_risk(cleaned)
        uncertainty = await self._estimate_uncertainty(cleaned)
        query_hash = __import__("hashlib").sha256(
            cleaned.encode("utf-8")
        ).hexdigest()[:16]

        # 模式建议
        if is_high_risk:
            suggested = "precision"
        elif uncertainty > 0.65:
            suggested = "precision"
        else:
            suggested = "fast"

        return PreprocessResult(
            cleaned_query=cleaned,
            keywords=keywords,
            intent=intent,
            uncertainty_score=uncertainty,
            suggested_mode=suggested,
            is_high_risk=is_high_risk,
            query_hash=query_hash,
        )

    def _detect_intent(self, query: str) -> str:
        """基于关键词规则识别用户意图"""
        query_lower = query.lower()
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            scores[intent] = sum(1 for p in patterns if p in query_lower)
        if not scores or max(scores.values()) == 0:
            return "factual"
        return max(scores, key=scores.get)

    def _check_high_risk(self, query: str) -> bool:
        """检查是否为高风险领域查询"""
        return any(kw in query for kw in self.HIGH_RISK_KEYWORDS)

    async def _estimate_uncertainty(self, query: str) -> float:
        """
        通过模型 logits 熵值估计答案不确定性。
        如果 LLM 不可用，退化为基于查询复杂度的启发式方法。
        """
        if self.llm:
            try:
                messages = [{"role": "user", "content": query}]
                return await self.llm.get_logits_uncertainty(messages)
            except Exception:
                pass

        # 启发式退化：查询越长、越复杂，不确定性越高
        if len(query) > 100:
            return 0.7
        elif len(query) > 50:
            return 0.5
        return 0.3
