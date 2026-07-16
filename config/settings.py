"""
全局配置中心 —— 对应方案文档「阶段0：插件初始化与配置」
支持管理员预设 + 用户自定义，风险场景强制触发精准模式。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# -- 工作模式 --
class WorkMode(str, Enum):
    FAST = "fast"           # 快速模式：低风险、高时效
    PRECISION = "precision" # 精准模式：高风险、高精准

# -- 高风险领域（强制精准模式）--
HIGH_RISK_DOMAINS = [
    "金融", "医疗", "法律", "药品", "投资",
    "保险", "证券", "手术", "法规", "税务",
    "finance", "medical", "legal", "pharmaceutical",
]

@dataclass
class PluginConfig:
    """插件全局配置，贯穿阶段0-4全部流程。"""

    # -- 阶段0：插件接入与模式配置 --
    default_mode: WorkMode = WorkMode.FAST
    force_precision_for_high_risk: bool = True
    admin_preset_mode: Optional[WorkMode] = None

    # -- 阶段1：预处理 --
    enable_uncertainty_detection: bool = True
    uncertainty_threshold: float = 0.65  # logits 熵值阈值，超过则建议精准模式

    # -- 阶段2：缓存 --
    cache_ttl_days: int = 30
    fast_mode_confidence_threshold: float = 0.70   # 缓存匹配置信度阈值
    precision_mode_confidence_threshold: float = 0.85

    # -- 阶段2：多源检索 --
    search_engines: list = field(default_factory=lambda: ["duckduckgo", "bing", "serpapi"])
    search_timeout_sec: int = 5
    max_search_results_per_engine: int = 8

    # -- 阶段2：事实验证 --
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"
    fact_verification_threshold: float = 0.75
    source_priority_order: list = field(default_factory=lambda: [
        "官方", "学术", "权威媒体", "普通资讯"
    ])

    # -- 阶段3：离线学习 --
    offline_learning_enabled: bool = True
    offline_learning_interval_hours: int = 168  # 每周
    high_quality_fact_threshold: float = 0.90   # 高质量事实筛选阈值
    lora_rank: int = 16
    lora_alpha: int = 32

    # -- 阶段4：用户交互 --
    enable_streaming_progress: bool = True
    enable_dynamic_mode_switch: bool = True
    enable_cross_session_cache: bool = True

    # -- 向量数据库 --
    vector_db_type: str = "chromadb"
    vector_db_path: str = "./data/vector_cache"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # -- 持久化路径 --
    data_dir: str = "./data"
    cache_db_dir: str = "./data/cache_db"
    feedback_log_path: str = "./data/feedback.jsonl"
    fact_library_path: str = "./data/verified_facts.jsonl"

    def get_effective_mode(self, user_query: str, user_choice: Optional[WorkMode] = None) -> WorkMode:
        """根据预设、用户选择、风险领域综合决策工作模式。"""
        if user_choice is not None:
            return user_choice
        if self.admin_preset_mode is not None:
            return self.admin_preset_mode
        if self.force_precision_for_high_risk and self._is_high_risk(user_query):
            return WorkMode.PRECISION
        return self.default_mode

    def _is_high_risk(self, query: str) -> bool:
        return any(domain in query for domain in HIGH_RISK_DOMAINS)

# 全局单例
config = PluginConfig()
