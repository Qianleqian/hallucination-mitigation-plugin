"""
用户交互增强 —— 对应方案文档「阶段4：用户交互增强（可选优化）」
实现: 实时进度反馈、动态模式切换、跨会话缓存复用、WebSocket 流式推送
"""
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from config.settings import config


@dataclass
class ProgressEvent:
    """进度事件"""
    stage: str        # preprocess / cache_check / search / verify / generate
    message: str
    progress_pct: float  # 0-100
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "message": self.message,
            "progress_pct": self.progress_pct,
            "timestamp": self.timestamp or datetime.now().isoformat(),
        }


class ProgressTracker:
    """
    实时进度追踪器。
    精准模式各阶段的进度百分比:
    - 预处理: 0-5%
    - 缓存检查: 5-10%
    - 查询改写: 10-20%
    - 多源搜索: 20-60%
    - 事实验证: 60-85%
    - 答案生成: 85-100%
    """
    STAGE_PROGRESS = {
        "preprocess":    (0, 5),
        "cache_check":   (5, 10),
        "query_rewrite": (10, 20),
        "search":        (20, 60),
        "verify":        (60, 85),
        "generate":      (85, 100),
    }

    def __init__(self):
        self._current_stage = ""
        self._callbacks: list[Callable] = []

    def on_progress(self, callback: Callable):
        """注册进度回调"""
        self._callbacks.append(callback)

    async def update(self, stage: str, message: str, sub_pct: float = 0.5):
        """
        更新进度。
        sub_pct: 当前阶段内的进度 (0-1)
        """
        if stage not in self.STAGE_PROGRESS:
            return

        self._current_stage = stage
        start, end = self.STAGE_PROGRESS[stage]
        overall_pct = start + (end - start) * sub_pct

        event = ProgressEvent(
            stage=stage,
            message=message,
            progress_pct=round(overall_pct, 1),
        )

        for cb in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception:
                pass


class DynamicModeSwitch:
    """
    动态模式切换。
    支持用户在精准模式执行过程中随时降级切换至快速模式。
    """

    def __init__(self):
        self._switch_requested = False
        self._target_mode = None

    def request_switch(self, target_mode: str):
        """请求切换模式"""
        self._switch_requested = True
        self._target_mode = target_mode

    def is_switch_requested(self) -> bool:
        return self._switch_requested

    def consume_switch(self) -> Optional[str]:
        """消费切换请求（调用后重置）"""
        if self._switch_requested:
            mode = self._target_mode
            self._switch_requested = False
            self._target_mode = None
            return mode
        return None

    def cancel_switch(self):
        """取消切换"""
        self._switch_requested = False
        self._target_mode = None


class FeedbackCollector:
    """
    用户反馈收集器。
    记录用户对回答的评价，作为离线学习的数据源之一。
    """

    def __init__(self, log_path: str = None):
        self.log_path = log_path or config.feedback_log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def record(self, query: str, answer: str, rating: str,
               comment: str = "", mode: str = ""):
        """记录一条用户反馈"""
        entry = {
            "query": query,
            "answer": answer[:500],
            "rating": rating,    # "good" / "neutral" / "bad"
            "comment": comment,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_stats(self) -> dict:
        """获取反馈统计"""
        if not os.path.exists(self.log_path):
            return {"total": 0, "good": 0, "neutral": 0, "bad": 0}

        stats = {"total": 0, "good": 0, "neutral": 0, "bad": 0}
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    stats["total"] += 1
                    rating = entry.get("rating", "neutral")
                    if rating in stats:
                        stats[rating] += 1
        except Exception:
            pass
        return stats

    def get_good_samples(self, limit: int = 100) -> list:
        """获取好评样本（用于离线学习）"""
        samples = []
        if not os.path.exists(self.log_path):
            return samples
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(samples) >= limit:
                        break
                    entry = json.loads(line.strip())
                    if entry.get("rating") == "good":
                        samples.append(entry)
        except Exception:
            pass
        return samples


class WebSocketProgressServer:
    """
    WebSocket 进度推送服务（阶段4）。
    通过 WebSocket 向客户端实时推送精准模式的处理进度。
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients = set()
        self._server = None

    async def start(self):
        """启动 WebSocket 服务"""
        try:
            import websockets
        except ImportError:
            print("[WebSocket] 请安装 websockets: pip install websockets")
            return

        async def handler(websocket):
            self._clients.add(websocket)
            try:
                async for message in websocket:
                    # 接收客户端指令（模式切换等）
                    data = json.loads(message)
                    if data.get("action") == "switch_mode":
                        await websocket.send(json.dumps({
                            "type": "mode_switched",
                            "mode": data.get("mode", "fast"),
                        }))
            except Exception:
                pass
            finally:
                self._clients.discard(websocket)

        self._server = await websockets.serve(handler, self.host, self.port)
        print(f"[WebSocket] 服务已启动: ws://{self.host}:{self.port}")

    async def broadcast(self, event: dict):
        """向所有连接的客户端广播进度"""
        if not self._clients:
            return
        message = json.dumps(event, ensure_ascii=False)
        # 使用 asyncio.gather 同步广播
        disconnected = set()
        for ws in self._clients:
            try:
                await ws.send(message)
            except Exception:
                disconnected.add(ws)
        self._clients -= disconnected

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# 全局单例
progress_tracker = ProgressTracker()
mode_switch = DynamicModeSwitch()
feedback_collector = FeedbackCollector()
ws_server = WebSocketProgressServer()
