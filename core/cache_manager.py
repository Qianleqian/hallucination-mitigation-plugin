"""
本地事实缓存管理 —— 对应方案文档「阶段2：缓存复用」
基于 Chroma/Qdrant 向量数据库实现知识缓存，支持 TTL 时效淘汰。
同时实现跨会话缓存复用（阶段4）。
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config.settings import config


@dataclass
class CacheEntry:
    """缓存条目"""
    id: str
    query: str
    answer: str
    confidence: float          # 匹配置信度
    source_urls: List[str] = field(default_factory=list)
    verified: bool = False     # 是否经过事实验证
    created_at: float = field(default_factory=time.time)
    ttl_days: int = 30
    access_count: int = 0

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_days * 86400

    def is_valid(self, threshold: float = 0.7) -> bool:
        return self.confidence >= threshold and not self.is_expired()


class CacheManager:
    """
    向量缓存管理器。
    快速模式阈值 0.70，精准模式阈值 0.85。

    支持：
    - 向量相似度匹配
    - TTL 时效淘汰
    - 跨会话复用
    - 持久化到磁盘
    """

    def __init__(self):
        self._cache: List[CacheEntry] = []
        self._embedder = None
        self._embeddings: List[np.ndarray] = []
        self._db_path = config.cache_db_dir
        self._load_from_disk()

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(config.embedding_model)
            except Exception:
                print("[缓存] 警告：无法加载嵌入模型，使用关键词匹配降级方案")
                self._embedder = None
        return self._embedder

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.7,
    ) -> Optional[CacheEntry]:
        """
        检索最佳匹配缓存。
        优先使用语义向量匹配；不可用时降级为关键词匹配。
        """
        if not self._cache:
            return None

        if self.embedder is not None:
            return self._semantic_search(query, top_k, threshold)
        else:
            return self._keyword_search(query, threshold)

    def _semantic_search(
        self, query: str, top_k: int, threshold: float
    ) -> Optional[CacheEntry]:
        """基于嵌入向量的语义匹配"""
        try:
            query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]

            if not self._embeddings:
                self._rebuild_embeddings()

            embeddings = np.array(self._embeddings)
            similarities = np.dot(embeddings, query_embedding)

            # 取 top_k
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            for idx in top_indices:
                score = float(similarities[idx])
                entry = self._cache[idx]
                if score >= threshold and entry.is_valid(threshold):
                    entry.access_count += 1
                    return entry

            return None
        except Exception as e:
            print(f"[缓存] 语义搜索异常: {e}")
            return self._keyword_search(query, threshold)

    def _keyword_search(self, query: str, threshold: float) -> Optional[CacheEntry]:
        """基于关键词重叠的匹配（降级方案）"""
        from utils.text_utils import extract_keywords, is_similar_query

        best_entry, best_score = None, 0.0
        for entry in self._cache:
            if not entry.is_valid(threshold):
                continue
            if is_similar_query(query, entry.query, threshold):
                score = len(set(extract_keywords(query)) &
                          set(extract_keywords(entry.query)))
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_entry:
            best_entry.access_count += 1
        return best_entry

    def store(
        self,
        query: str,
        answer: str,
        confidence: float = 0.8,
        source_urls: List[str] = None,
        verified: bool = False,
        ttl_days: int = None,
    ):
        """存储新的缓存条目"""
        entry = CacheEntry(
            id=f"cache_{int(time.time() * 1000)}_{len(self._cache)}",
            query=query,
            answer=answer,
            confidence=confidence,
            source_urls=source_urls or [],
            verified=verified,
            ttl_days=ttl_days or config.cache_ttl_days,
        )
        self._cache.append(entry)

        # 增量更新嵌入
        if self.embedder is not None:
            try:
                emb = self.embedder.encode([query], normalize_embeddings=True)[0]
                self._embeddings.append(emb)
            except Exception:
                pass

        # 定期持久化
        if len(self._cache) % 50 == 0:
            self._save_to_disk()

    def evict_expired(self) -> int:
        """淘汰过期缓存"""
        before = len(self._cache)
        valid_entries = [(e, self._embeddings[i] if i < len(self._embeddings) else None)
                         for i, e in enumerate(self._cache) if not e.is_expired()]
        if valid_entries:
            self._cache = [ve[0] for ve in valid_entries]
            self._embeddings = [ve[1] for ve in valid_entries if ve[1] is not None]
        else:
            self._cache = []
            self._embeddings = []
        return before - len(self._cache)

    def get_high_quality_entries(self, threshold: float = 0.9) -> List[CacheEntry]:
        """获取高质量缓存条目（阶段3：离线学习数据源）"""
        return [e for e in self._cache
                if e.confidence >= threshold and e.verified and not e.is_expired()]

    def _rebuild_embeddings(self):
        """重建所有嵌入"""
        if self.embedder is None:
            self._embeddings = []
            return
        try:
            queries = [e.query for e in self._cache]
            self._embeddings = self.embedder.encode(
                queries, normalize_embeddings=True
            ).tolist()
        except Exception:
            self._embeddings = []

    def _save_to_disk(self):
        """持久化到磁盘（跨会话复用）"""
        try:
            os.makedirs(self._db_path, exist_ok=True)
            path = os.path.join(self._db_path, "cache_entries.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for entry in self._cache[-500:]:  # 保留最近 500 条
                    f.write(json.dumps({
                        "id": entry.id,
                        "query": entry.query,
                        "answer": entry.answer,
                        "confidence": entry.confidence,
                        "source_urls": entry.source_urls,
                        "verified": entry.verified,
                        "created_at": entry.created_at,
                        "ttl_days": entry.ttl_days,
                    }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[缓存] 持久化失败: {e}")

    def _load_from_disk(self):
        """从磁盘恢复缓存（跨会话复用）"""
        try:
            path = os.path.join(self._db_path, "cache_entries.jsonl")
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entry = CacheEntry(
                        id=data.get("id", ""),
                        query=data.get("query", ""),
                        answer=data.get("answer", ""),
                        confidence=data.get("confidence", 0.8),
                        source_urls=data.get("source_urls", []),
                        verified=data.get("verified", False),
                        created_at=data.get("created_at", time.time()),
                        ttl_days=data.get("ttl_days", 30),
                    )
                    if not entry.is_expired():
                        self._cache.append(entry)
            print(f"[缓存] 从磁盘恢复 {len(self._cache)} 条缓存")
        except Exception as e:
            print(f"[缓存] 磁盘恢复失败: {e}")

    @property
    def stats(self) -> dict:
        return {
            "total_entries": len(self._cache),
            "verified_count": sum(1 for e in self._cache if e.verified),
            "expired_count": sum(1 for e in self._cache if e.is_expired()),
            "avg_confidence": (
                sum(e.confidence for e in self._cache) / len(self._cache)
                if self._cache else 0
            ),
        }
