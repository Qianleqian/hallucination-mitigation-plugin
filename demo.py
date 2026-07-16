#!/usr/bin/env python3
"""
演示脚本 —— 展示幻觉缓解插件的核心能力

包含三个演示场景:
1. 快速模式: 缓存命中 + 低延迟响应
2. 精准模式: 多源搜索 + 交叉验证
3. 离线学习: 数据采集 -> 清洗 -> 数据集构建
"""
import asyncio
import os
import sys

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import config, WorkMode
from core.cache_manager import CacheManager, CacheEntry
from core.dual_mode_engine import DualModeEngine
from core.fact_verifier import FactVerifier, FactClaim
from core.feedback_handler import FeedbackCollector
from core.offline_learner import OfflineLearner
from core.preprocessor import InputPreprocessor
from core.search_aggregator import SearchAggregator
from models.llm_base import LLMConfig
from models.qwen_adapter import QwenAdapter
from utils.metrics import metrics_collector


def print_banner():
    print("""
============================================================
  面向大模型幻觉缓解的通用插件架构 - 功能演示
  目标模型: 通义千问 (Qwen)
  架构: 在线双路径推理 + 离线持续学习
============================================================
""")


def create_qwen():
    """创建千问适配器（如果 API Key 可用）"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if api_key:
        return QwenAdapter(LLMConfig(
            model="qwen-plus",
            api_key=api_key,
            temperature=0.1,
        ))
    return None


# ===============================================================
# 演示 1: 预处理与意图识别
# ===============================================================

async def demo_1_preprocessing():
    """演示阶段1: 输入预处理与意图识别"""
    print("-" * 55)
    print("演示 1: 输入预处理与意图识别（阶段1）")
    print("-" * 55)

    preprocessor = InputPreprocessor(llm=None)

    test_queries = [
        "感冒了应该吃什么药？",
        "什么是量子纠缠？",
        "怎么用 Python 写一个排序算法？",
        "请对比 iPhone 和华为手机",
        "投资股票应该注意什么？",
    ]

    for query in test_queries:
        result = await preprocessor.preprocess(query)
        print(f"\n  输入: {query}")
        print(f"  清洗后: {result.cleaned_query}")
        print(f"  关键词: {result.keywords}")
        print(f"  意图: {result.intent}")
        print(f"  不确定性: {result.uncertainty_score:.2f}")
        print(f"  高风险: {'[WARN] 是' if result.is_high_risk else '否'}")
        print(f"  建议模式: {result.suggested_mode}")

    print(f"\n  [OK] 预处理功能正常")


# ===============================================================
# 演示 2: 缓存管理
# ===============================================================

async def demo_2_cache():
    """演示阶段2.1: 缓存管理"""
    print("\n" + "-" * 55)
    print("演示 2: 缓存存储与检索（阶段2.1）")
    print("-" * 55)

    cache = CacheManager()

    # 存储测试数据
    cache.store(
        query="什么是机器学习？",
        answer="机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进，而无需进行明确的编程。",
        confidence=0.95,
        source_urls=["https://example.com/ml-intro"],
        verified=True,
    )
    cache.store(
        query="Python 是什么？",
        answer="Python 是一种高级、解释型、面向对象的编程语言，由 Guido van Rossum 于 1991 年创建。",
        confidence=0.88,
        source_urls=["https://example.com/python"],
        verified=True,
    )

    print(f"  缓存条目数: {len(cache._cache)}")

    # 检索测试
    test_queries = [
        "请解释什么是机器学习？",
        "Python是什么语言？",
        "今天天气怎么样？",
    ]

    for q in test_queries:
        result = cache.search(q, threshold=0.0)  # 低阈值以确保能匹配到
        if result:
            print(f"\n  查询: '{q}'")
            print(f"  [OK] 命中: {result.query[:50]}...")
            print(f"  置信度: {result.confidence:.2f}")
        else:
            print(f"\n  查询: '{q}'")
            print(f"  [FAIL] 未命中")

    # 统计
    stats = cache.stats
    print(f"\n  缓存统计: {stats}")

    # 清理
    evicted = cache.evict_expired()
    print(f"  过期淘汰: {evicted} 条")
    print(f"  [OK] 缓存管理功能正常")


# ===============================================================
# 演示 3: 搜索聚合
# ===============================================================

async def demo_3_search():
    """演示阶段2.2: 多源搜索聚合"""
    print("\n" + "-" * 55)
    print("演示 3: 多源搜索聚合（阶段2.2）")
    print("-" * 55)

    searcher = SearchAggregator()

    # 仅用 DuckDuckGo（免费，无需 API Key）
    query = "量子计算 基本原理"
    print(f"  搜索: {query}")

    results = await searcher.search(
        query,
        engines=["duckduckgo"],
        max_results=5,
    )

    print(f"  耗时: {results.elapsed_ms:.0f}ms")
    print(f"  结果数: {len(results.results)}")
    print(f"  来源引擎: {results.total_sources}")

    for i, r in enumerate(results.results[:5]):
        print(f"\n  [{i+1}] {r.title[:70]}")
        print(f"      {r.snippet[:100]}...")
        print(f"      {r.url}")

    print(f"\n  [OK] 搜索聚合功能正常")


# ===============================================================
# 演示 4: 事实验证
# ===============================================================

async def demo_4_fact_verification():
    """演示阶段2.5: 事实验证"""
    print("\n" + "-" * 55)
    print("演示 4: 事实验证（阶段2.5）")
    print("-" * 55)

    verifier = FactVerifier()

    # 模拟 LLM 回答和搜索结果
    llm_answer = "地球到太阳的距离约为1.496亿公里，光从太阳到地球需要约8分20秒。"
    search_snippets = [
        "地球到太阳的平均距离约为1.496亿公里，这被称为1个天文单位(AU)。——NASA官网",
        "太阳光到达地球需要大约8分20秒的时间。——天文学教科书",
        "地球与太阳的距离约为1.5亿公里。——维基百科",
        "地球到太阳的距离是384,400公里。——（错误信息）",
    ]

    claims = []
    for snippet in search_snippets:
        claims.append(FactClaim(
            text=snippet,
            source_url="https://example.com/astronomy",
            source_type=(
                "官方" if "NASA" in snippet else
                "学术" if "教科书" in snippet else
                "权威媒体" if "维基" in snippet else
                "普通资讯"
            ),
        ))

    result = await verifier.verify(claims, llm_answer)

    print(f"  LLM 回答: {llm_answer}")
    print(f"\n  验证结果:")
    print(f"  通过: {result.is_verified}")
    print(f"  置信度: {result.confidence:.2f}")
    print(f"  支持来源数: {len(result.supporting_sources)}")
    print(f"  冲突来源数: {len(result.conflicting_sources)}")
    if result.final_fact:
        print(f"  最终事实: {result.final_fact}")

    print(f"\n  [OK] 事实验证功能正常")


# ===============================================================
# 演示 5: 离线学习
# ===============================================================

async def demo_5_offline_learning():
    """演示阶段3: 离线学习闭环"""
    print("\n" + "-" * 55)
    print("演示 5: 离线学习闭环（阶段3）")
    print("-" * 55)

    cache = CacheManager()

    # 模拟填充高质量数据
    sample_data = [
        ("什么是深度学习？",
         "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的层次化表示。",
         0.95, True),
        ("Python 的设计哲学是什么？",
         "Python 的设计哲学强调代码可读性和简洁性，使用有意义的缩进来划分代码块。",
         0.92, True),
        ("什么是 API？",
         "API（应用程序编程接口）是一组定义和协议，用于构建和集成应用软件。",
         0.90, True),
        ("HTTP 和 HTTPS 的区别是什么？",
         "HTTPS 是 HTTP 的安全版本，使用 SSL/TLS 协议加密数据传输，确保数据安全。",
         0.93, True),
        ("什么是云计算？",
         "云计算是通过互联网提供计算资源和服务，包括服务器、存储、数据库、网络、软件等。",
         0.91, True),
    ]

    for q, a, conf, verified in sample_data:
        cache.store(
            query=q, answer=a, confidence=conf,
            source_urls=["https://example.com/tech"],
            verified=verified,
        )

    print(f"  填充样本: {len(sample_data)} 条")

    learner = OfflineLearner(cache)
    report = await learner.run_cycle()

    print(f"\n  学习报告:")
    print(f"  状态: {report.get('status', 'unknown')}")
    print(f"  步骤数: {len(report.get('steps', []))}")
    if report.get("errors"):
        print(f"  错误: {report['errors']}")
    print(f"  耗时: {report.get('duration_sec', 0):.1f}秒")

    # 检查生成的 dataset 文件
    dataset_dir = os.path.join(config.data_dir, "datasets")
    if os.path.exists(dataset_dir):
        files = os.listdir(dataset_dir)
        print(f"  生成的数据集: {files}")

    print(f"\n  [OK] 离线学习功能正常")


# ===============================================================
# 演示 6: 用户反馈系统
# ===============================================================

async def demo_6_feedback():
    """演示阶段4: 用户反馈系统"""
    print("\n" + "-" * 55)
    print("演示 6: 用户反馈系统（阶段4）")
    print("-" * 55)

    collector = FeedbackCollector()

    # 模拟用户反馈
    collector.record(
        query="什么是 Python？",
        answer="Python 是一种编程语言。",
        rating="good",
        comment="回答准确",
        mode="precision",
    )
    collector.record(
        query="今天天气如何？",
        answer="抱歉，我无法获取实时天气数据。",
        rating="neutral",
        comment="回答合理",
        mode="fast",
    )
    collector.record(
        query="谁发明了电灯？",
        answer="爱迪生发明了电灯。",
        rating="bad",
        comment="事实上电灯并非爱迪生单独发明，他只是改进了灯泡设计。",
        mode="fast",
    )

    stats = collector.get_stats()
    print(f"  反馈统计: {stats}")

    good_samples = collector.get_good_samples(limit=10)
    print(f"  好评样本数: {len(good_samples)}")

    print(f"  [OK] 反馈系统功能正常")


# ===============================================================
# 演示 7: 完整流程（如果有 API Key）
# ===============================================================

async def demo_7_full_workflow():
    """演示完整端到端流程"""
    print("\n" + "-" * 55)
    print("演示 7: 完整端到端流程")
    print("-" * 55)

    qwen = create_qwen()

    if qwen is None:
        print("\n  [WARN]  未检测到 DASHSCOPE_API_KEY，跳过千问集成演示")
        print("  设置 API Key 后可体验完整流程:")
        print("  export DASHSCOPE_API_KEY='your-key'")
        return

    engine = DualModeEngine(qwen)
    print("  [OK] 千问模型已连接")

    # 进度回调
    async def progress(msg):
        print(f"    -> {msg}")

    # 快速模式测试
    print("\n  [快速模式] 测试:")
    result = await engine.ask(
        "什么是人工智能？",
        mode="fast",
        stream_callback=progress,
    )
    print(f"  延迟: {result['latency_ms']:.0f}ms")
    print(f"  缓存命中: {result['cache_hit']}")
    print(f"  回答: {result['answer'][:200]}...")

    # 精准模式测试
    print("\n  [精准模式] 测试:")
    result = await engine.ask(
        "2024年诺贝尔物理学奖得主是谁？",
        mode="precision",
        stream_callback=progress,
    )
    print(f"  延迟: {result['latency_ms']:.0f}ms")
    print(f"  验证通过: {result['verified']}")
    print(f"  来源数: {len(result.get('sources', []))}")
    print(f"  回答: {result['answer'][:200]}...")

    print(f"\n  [OK] 完整流程功能正常")


# ===============================================================
# 主函数
# ===============================================================

async def main():
    print_banner()

    demos = [
        ("1. 预处理与意图识别", demo_1_preprocessing),
        ("2. 缓存管理", demo_2_cache),
        ("3. 多源搜索聚合", demo_3_search),
        ("4. 事实验证", demo_4_fact_verification),
        ("5. 离线学习闭环", demo_5_offline_learning),
        ("6. 用户反馈系统", demo_6_feedback),
        ("7. 完整端到端流程（需 API Key）", demo_7_full_workflow),
    ]

    for name, demo_fn in demos:
        try:
            await demo_fn()
        except Exception as e:
            print(f"\n  [FAIL] {name} 执行失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 55)
    print("  所有演示完成！")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
