#!/usr/bin/env python3
"""
面向大模型幻觉缓解的通用插件架构 —— 主入口

基于方案文档「在线双路径推理 + 离线持续学习」架构，
以通义千问 (Qwen) 为插件目标模型。

使用:
    # 设置 API Key
    export DASHSCOPE_API_KEY="your-dashscope-api-key"

    # 命令行交互
    python main.py

    # 指定模式
    python main.py --mode precision
    python main.py --mode fast

    # 单次提问
    python main.py --query "什么是量子计算？"

    # 执行离线学习
    python main.py --offline-learn

    # 启动 WebSocket 服务
    python main.py --serve
"""
import argparse
import asyncio
import os
import sys

from config.settings import config, WorkMode
from core.dual_mode_engine import DualModeEngine
from core.offline_learner import OfflineLearner
from core.feedback_handler import (
    progress_tracker, mode_switch, feedback_collector, ws_server,
)
from models.llm_base import LLMConfig
from models.qwen_adapter import QwenAdapter
from utils.metrics import metrics_collector


def create_qwen_llm(model: str = "qwen-plus"):
    """创建千问适配器"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("=" * 60)
        print("[WARN]  请设置 DASHSCOPE_API_KEY 环境变量")
        print("   export DASHSCOPE_API_KEY='your-key'")
        print("   获取 Key: https://dashscope.console.aliyun.com/")
        print("=" * 60)
        sys.exit(1)

    return QwenAdapter(LLMConfig(
        model=model,
        api_key=api_key,
        temperature=0.1,
        max_tokens=2048,
    ))


async def interactive_mode(engine: DualModeEngine):
    """交互式命令行模式"""
    print("=" * 60)
    print("  大模型幻觉缓解插件系统 - 交互模式")
    print("  目标模型: 通义千问 (Qwen)")
    print("  当前模式: 自动 (高风险自动切换精准模式)")
    print("=" * 60)
    print("  命令:")
    print("    /fast      切换至快速模式（当前会话）")
    print("    /precision 切换至精准模式（当前会话）")
    print("    /stats     查看会话统计")
    print("    /feedback  对上一次回答评分 (good/neutral/bad)")
    print("    /exit      退出")
    print("=" * 60)

    current_mode = None
    last_result = None

    while True:
        try:
            mode_label = current_mode or "auto"
            user_input = input(f"\n[{mode_label}] 请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith("/"):
            cmd = user_input[1:].lower()
            if cmd == "exit":
                print("再见！")
                break
            elif cmd == "fast":
                current_mode = "fast"
                print("已切换至快速模式")
                continue
            elif cmd == "precision":
                current_mode = "precision"
                print("已切换至精准模式")
                continue
            elif cmd == "stats":
                print(metrics_collector.get_summary())
                continue
            elif cmd == "feedback":
                if last_result:
                    rating = input("请评分 (good/neutral/bad): ").strip()
                    feedback_collector.record(
                        query=user_input,
                        answer=last_result.get("answer", ""),
                        rating=rating,
                        mode=last_result.get("mode", ""),
                    )
                    print("感谢反馈！")
                else:
                    print("暂无问答记录")
                continue
            else:
                print(f"未知命令: {cmd}")
                continue

        # 进度回调
        async def on_progress(msg: str):
            print(f"  ... {msg}")

        # 执行问答
        print(f"  模式: {current_mode or '自动'}")
        result = await engine.ask(
            user_input,
            mode=current_mode,
            stream_callback=on_progress,
        )
        last_result = result

        print(f"\n{'-' * 50}")
        print(f">> 回答 ({result['mode']} | {result['latency_ms']:.0f}ms):")
        print(f"{result['answer']}")
        print(f"{'-' * 50}")
        print(f"  置信度: {result.get('confidence', 0):.2f}")
        print(f"  缓存命中: {result.get('cache_hit', False)}")
        print(f"  事实验证: {result.get('verified', False)}")
        if result.get("sources"):
            print(f"  来源数: {len(result['sources'])}")


async def single_query(engine: DualModeEngine, query: str, mode: str = None):
    """单次查询模式"""
    async def on_progress(msg: str):
        print(f"  ... {msg}")

    result = await engine.ask(query, mode=mode, stream_callback=on_progress)
    print(f"\n模式: {result['mode']} | 延迟: {result['latency_ms']:.0f}ms")
    print(f"回答: {result['answer']}")
    return result


async def run_offline_learning(engine: DualModeEngine):
    """执行离线学习周期"""
    print("开始离线学习周期...")
    learner = OfflineLearner(engine.cache)
    report = await learner.run_cycle()
    print(f"\n离线学习报告:")
    for step in report.get("steps", []):
        print(f"  [OK] {step}")
    print(f"状态: {report.get('status', 'unknown')}")
    print(f"耗时: {report.get('duration_sec', 0):.1f}秒")


async def main():
    parser = argparse.ArgumentParser(
        description="大模型幻觉缓解通用插件 - 千问版"
    )
    parser.add_argument(
        "--mode", choices=["fast", "precision", "auto"],
        default=None,
        help="工作模式（默认自动）"
    )
    parser.add_argument(
        "--query", type=str,
        help="单次提问模式"
    )
    parser.add_argument(
        "--offline-learn", action="store_true",
        help="执行离线学习周期"
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="启动 WebSocket 服务"
    )
    parser.add_argument(
        "--model", type=str, default="qwen-plus",
        choices=QwenAdapter.AVAILABLE_MODELS,
        help="千问模型版本（默认 qwen-plus）"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="显示会话统计"
    )
    args = parser.parse_args()

    # 初始化千问
    llm = create_qwen_llm(model=args.model)
    engine = DualModeEngine(llm)
    print(f"[OK] 千问模型已连接: {args.model}")

    if args.serve:
        # 启动 WebSocket 服务
        await ws_server.start()
        print("WebSocket 服务运行中，按 Ctrl+C 停止...")
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            await ws_server.stop()

    elif args.offline_learn:
        await run_offline_learning(engine)

    elif args.query:
        await single_query(engine, args.query, mode=args.mode)

    elif args.stats:
        print(metrics_collector.get_summary())

    else:
        # 默认: 交互模式
        await interactive_mode(engine)

    # 退出前显示统计
    summary = metrics_collector.get_summary()
    if summary:
        print(f"\n会话统计: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
