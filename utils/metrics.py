"""评估指标"""
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ResponseMetrics:
    """每次问答的指标收集"""
    query: str
    mode: str                          # fast / precision
    response: str
    latency_ms: float
    cache_hit: bool = False
    sources: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    fact_verification_passed: bool = False
    user_feedback: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "mode": self.mode,
            "response": self.response[:500],
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "sources": self.sources,
            "confidence_score": self.confidence_score,
            "fact_verification_passed": self.fact_verification_passed,
            "user_feedback": self.user_feedback,
        }


class MetricsCollector:
    """全局指标收集器"""

    def __init__(self):
        self.records: List[ResponseMetrics] = []
        self._session_start = time.time()

    def record(self, metrics: ResponseMetrics):
        self.records.append(metrics)

    def get_summary(self) -> dict:
        if not self.records:
            return {}
        fast = [r for r in self.records if r.mode == "fast"]
        precision = [r for r in self.records if r.mode == "precision"]
        total = len(self.records)

        def avg_latency(lst):
            return sum(r.latency_ms for r in lst) / len(lst) if lst else 0

        def accuracy(lst):
            # 准确率 = 通过验证 / 总数
            if not lst:
                return 0
            verified = [r for r in lst if r.fact_verification_passed]
            return len(verified) / len(lst)

        return {
            "total_queries": total,
            "session_duration_sec": time.time() - self._session_start,
            "fast_mode": {
                "count": len(fast),
                "avg_latency_ms": avg_latency(fast),
                "cache_hit_rate": sum(1 for r in fast if r.cache_hit) / len(fast) if fast else 0,
            },
            "precision_mode": {
                "count": len(precision),
                "avg_latency_ms": avg_latency(precision),
                "accuracy": accuracy(precision),
            },
        }


# 全局单例
metrics_collector = MetricsCollector()
