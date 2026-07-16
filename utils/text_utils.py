"""文本处理工具"""
import re
import hashlib
from typing import List, Optional


def clean_text(text: str) -> str:
    """数据清洗与格式标准化（阶段1预处理）"""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def compute_text_hash(text: str) -> str:
    """计算文本哈希用于缓存去重"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """简单关键词抽取：按词频 + 停用词过滤"""
    stop_words = {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
    }
    words = re.findall(r"[一-鿿]+|[a-zA-Z]+", text.lower())
    freq = {}
    for w in words:
        if w not in stop_words and len(w) > 1:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]


def is_similar_query(text1: str, text2: str, threshold: float = 0.75) -> bool:
    """基于关键词重叠率判断两个查询是否相似（跨会话缓存复用）"""
    kw1 = set(extract_keywords(text1, top_n=10))
    kw2 = set(extract_keywords(text2, top_n=10))
    if not kw1 or not kw2:
        return False
    intersection = kw1 & kw2
    return len(intersection) / min(len(kw1), len(kw2)) >= threshold


def format_uncertainty_disclaimer(confidence_level: str) -> str:
    """生成标准免责声明"""
    disclaimers = {
        "low":    "\n\n---\n> [WARN] 置信度：低 | 本回答基于模型已有知识生成，建议开启精准模式进行多源核验。",
        "medium": "\n\n---\n> [INFO] 置信度：中 | 本回答仅供参考，如需确认请使用精准模式获取多源验证信息。",
        "high":   "\n\n---\n> [PASS] 置信度：高 | 本回答已通过多源事实验证。",
    }
    return disclaimers.get(confidence_level, disclaimers["low"])


def split_into_sentences(text: str) -> List[str]:
    """分句"""
    return re.split(r"[。.！!？?\n]+", text)
