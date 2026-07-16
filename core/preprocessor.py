"""
输入预处理器 —— 对应方案文档「阶段1：用户输入与预处理」
完成数据清洗、意图识别、问题类型分类、不确定性预判、前提合理性检测。

新增: 问题类型分类 (factual / logic / chat)
     - 事实类 → precision (多源验证)
     - 逻辑/常识类 → logic_prompt (专用提示词，不走检索)
     - 闲聊类 → fast (缓存优先)
"""
import re
from dataclasses import dataclass, field
from typing import Optional, List

from models.llm_base import BaseLLM
from utils.text_utils import clean_text, extract_keywords


@dataclass
class PreprocessResult:
    """预处理结果"""
    cleaned_query: str
    keywords: list
    intent: str                    # 问题意图类型
    question_type: str = "factual" # 新增: factual / logic / chat
    is_logic_question: bool = False  # 新增: 是否为逻辑/常识类
    is_premise_trap: bool = False    # 新增: 问题前提是否可能有陷阱
    logic_indicators: List[str] = field(default_factory=list)  # 新增: 触发的逻辑关键词
    logic_prompt: Optional[str] = None  # 新增: 逻辑类专用提示词
    uncertainty_score: float = 0.3
    suggested_mode: str = "fast"
    is_high_risk: bool = False
    query_hash: str = ""


# 逻辑/常识问题分类关键词 (轻量级规则，无需 BERT)
LOGIC_PATTERNS = {
    # 物理不可能
    "physics_impossible": {
        "keywords": ["同时.*在.*和", "同时位于", "既能.*也能", "既在.*又在",
                      "同时处于", "同一时刻.*两地", "同时出现在",
                      "同一时间.*不同地点", "分身", "瞬时移动"],
        "label": "物理不可能",
    },
    # 数学/范畴矛盾
    "math_contradiction": {
        "keywords": ["既是.*又是", "奇数.*偶数", "同时等于", "既是偶数又是奇数",
                      "同时.*等于.*和", "既是正数又是负数",
                      r"\d+度.*同时.*\d+度", "无理数.*整数"],
        "label": "数学矛盾",
    },
    # 时间悖论
    "temporal_paradox": {
        "keywords": ["今天出生.*昨天", "去世后.*写", "未来.*记忆", "死后.*完成",
                      "还没出生.*就", "先.*后.*是否可能", "死后.*写了一本",
                      "年.*发明.*互联网.*189", r"\d+年的人.*\d+年的事"],
        "label": "时间悖论",
    },
    # 身份/关系矛盾
    "identity_contradiction": {
        "keywords": ["自己的父亲", "自己的母亲", "未婚.*鳏夫", "未婚.*寡妇",
                      "比.*年龄.*大.*岁.*儿.*岁", "自己.*自己.*关系",
                      "既是父亲又是", "单身.*已婚"],
        "label": "身份矛盾",
    },
    # 因果颠倒
    "causal_reversal": {
        "keywords": ["因为.*得了.*所以.*开始吸烟", "结果导致.*原因",
                      "因为.*治愈.*所以.*生病", "因果关系.*成立吗",
                      "因为.*死.*所以.*杀"],
        "label": "因果颠倒",
    },
    # 单位/范畴错误
    "category_error": {
        "keywords": ["摄氏度.*换算.*人民币", "公斤.*同时也重.*米",
                      "米.*等于.*千克", "颜色.*重量",
                      "温度.*等于.*长度"],
        "label": "范畴错误",
    },
    # 空间矛盾
    "spatial_contradiction": {
        "keywords": ["同时在北京和纽约", "南极.*官方语言",
                      "一个房间.*没有人.*同时有人", "空的和满的",
                      "同时开门和关门"],
        "label": "空间矛盾",
    },
    # 全称/存在矛盾
    "universal_contradiction": {
        "keywords": ["所有.*都是.*但存在", "每.*都.*但有一",
                      "没有.*同时有.*存在", "不存在.*但.*有",
                      "全部.*但是.*例外"],
        "label": "全称矛盾",
    },
    # 语义/定义冲突
    "definition_conflict": {
        "keywords": ["正方.*圆", "三角.*四边", "已婚.*单身",
                      "素食.*吃肉", "无色.*绿色"],
        "label": "定义冲突",
    },
}

# 事实类关键词
FACTUAL_PATTERNS = [
    "是什么", "什么是", "谁", "何时", "哪里", "多少", "怎么定义",
    "哪一年", "哪个国家", "多少公里", "多少度", "多高", "多重",
    "什么时候", "在哪里", "创始人", "发明者", "作者",
    "what is", "who is", "when", "where", "how many", "how much",
    "定义", "概念", "解释", "原理",
]

# 闲聊类关键词
CHAT_PATTERNS = [
    "你好", "谢谢", "再见", "哈哈", "讲个笑话", "聊天",
    "你今天", "你叫什么", "你是谁", "介绍一下你自己",
    "推荐.*电影", "推荐.*书", "推荐.*音乐",
    "hello", "hi", "thanks", "bye",
]

# 逻辑类专用系统提示词
LOGIC_SYSTEM_PROMPT = """你是一个逻辑判断专家。这是逻辑判断题，请遵守:

1. 用常识和逻辑直接作答，一句话给结论（"可以"/"不可以"/"矛盾"/"不矛盾"）
2. 不要展开理论讨论，不要引用量子力学、相对论等来回避简单判断
3. 如果问题前提自相矛盾，直接指出矛盾所在
4. 如果是非判断，结论放在第一句
5. 用中文回答，简洁直接"""


class InputPreprocessor:
    """
    阶段1 预处理器。
    新增: 问题类型分类 (factual / logic / chat)，前提合理性预判。
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

    HIGH_RISK_KEYWORDS = [
        "金融", "医疗", "法律", "药品", "手术", "投资",
        "证券", "保险", "税务", "法规", "剂量", "禁忌",
    ]

    def __init__(self, llm: Optional[BaseLLM] = None):
        self.llm = llm

    async def preprocess(self, raw_query: str) -> PreprocessResult:
        """入口：完成全部预处理（含问题类型分类）"""
        cleaned = clean_text(raw_query)
        keywords = extract_keywords(cleaned)
        intent = self._detect_intent(cleaned)
        is_high_risk = self._check_high_risk(cleaned)
        uncertainty = await self._estimate_uncertainty(cleaned)
        query_hash = __import__("hashlib").sha256(
            cleaned.encode("utf-8")
        ).hexdigest()[:16]

        # 新增: 问题类型分类
        question_type, is_logic, logic_indicators = self._classify_question_type(cleaned)

        # 新增: 前提合理性预判
        is_premise_trap = self._detect_premise_trap(cleaned, is_logic)

        # 新增: 逻辑类专用提示词
        logic_prompt = LOGIC_SYSTEM_PROMPT if is_logic else None

        # 模式决策: 逻辑类走专用提示词路径（不走检索）
        if is_logic:
            suggested = "logic"      # 新增第三种路径
        elif is_high_risk:
            suggested = "precision"
        elif uncertainty > 0.65:
            suggested = "precision"
        elif question_type == "chat":
            suggested = "fast"
        elif question_type == "factual":
            suggested = "precision"
        else:
            suggested = "fast"

        return PreprocessResult(
            cleaned_query=cleaned,
            keywords=keywords,
            intent=intent,
            question_type=question_type,
            is_logic_question=is_logic,
            is_premise_trap=is_premise_trap,
            logic_indicators=logic_indicators,
            logic_prompt=logic_prompt,
            uncertainty_score=uncertainty,
            suggested_mode=suggested,
            is_high_risk=is_high_risk,
            query_hash=query_hash,
        )

    # ═══════════════════════════════════════════════════════
    # 新增: 问题类型分类
    # ═══════════════════════════════════════════════════════

    def _classify_question_type(self, query: str) -> tuple:
        """
        轻量级三分类: factual / logic / chat
        返回: (类型, 是否逻辑类, 触发的逻辑关键词列表)
        """
        # 检查逻辑模式
        indicators = []
        for category, config in LOGIC_PATTERNS.items():
            for pattern in config["keywords"]:
                if re.search(pattern, query):
                    indicators.append(config["label"])
                    break  # 每个类别只记一次

        if indicators:
            return ("logic", True, indicators)

        # 检查闲聊模式
        for pattern in CHAT_PATTERNS:
            if re.search(pattern, query):
                return ("chat", False, [])

        # 检查事实模式
        for pattern in FACTUAL_PATTERNS:
            if pattern in query:
                return ("factual", False, [])

        # 默认: factual
        return ("factual", False, [])

    # ═══════════════════════════════════════════════════════
    # 新增: 前提合理性预判
    # ═══════════════════════════════════════════════════════

    def _detect_premise_trap(self, query: str, is_logic: bool) -> bool:
        """
        检测问题前提是否可能包含陷阱。
        陷阱特征: 前提假设了一个不存在/不合理的事实。
        """
        trap_patterns = [
            # 假设未来事件已发生
            r"202[5-9]年.*冠军.*是谁", r"20[3-9]\d年.*当选",
            r"明年.*已经", r"下届.*冠军",
            # 假设错误归属
            r"谁在\d{4}年首次登上火星", r"爱因斯坦.*诺贝尔化学",
            r"莎士比亚.*哈利波特", r"特斯拉.*发明.*爱迪生",
            # 不可能的前提
            r"去世后又写", r"今天出生.*昨天",
            r"同时.*在北京.*纽约", r"未婚.*鳏夫",
        ]
        if is_logic:
            return True  # 逻辑问题本身就有陷阱可能
        for pattern in trap_patterns:
            if re.search(pattern, query):
                return True
        return False

    # ═══════════════════════════════════════════════════════
    # 原有方法（保持不变）
    # ═══════════════════════════════════════════════════════

    def _detect_intent(self, query: str) -> str:
        query_lower = query.lower()
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            scores[intent] = sum(1 for p in patterns if p in query_lower)
        if not scores or max(scores.values()) == 0:
            return "factual"
        return max(scores, key=scores.get)

    def _check_high_risk(self, query: str) -> bool:
        return any(kw in query for kw in self.HIGH_RISK_KEYWORDS)

    async def _estimate_uncertainty(self, query: str) -> float:
        if self.llm:
            try:
                messages = [{"role": "user", "content": query}]
                return await self.llm.get_logits_uncertainty(messages)
            except Exception:
                pass
        if len(query) > 100:
            return 0.7
        elif len(query) > 50:
            return 0.5
        return 0.3
