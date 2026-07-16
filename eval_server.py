#!/usr/bin/env python3
"""
服务器端评估脚本 —— 用大量测试集评估千问幻觉缓解插件效果
"""
import asyncio
import json
import sys
import time
import urllib.request

sys.path.insert(0, ".")
from utils.test_datasets import (
    get_all_builtin_tests, DatasetDownloader,
    HallucinationEvaluator, TestCase, EvalResult, create_full_test_suite,
)

BASE_URL = "http://127.0.0.1:8009/v1"
MODEL = "qwen35-9b-thinking"


def qwen_ask(system_prompt: str, user_query: str, max_tokens=1000) -> tuple:
    """调用本地千问 API"""
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        d = json.loads(resp.read().decode())

    msg = d["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning", "") or ""
    latency = (time.time() - t0) * 1000
    return content.strip(), latency


def run_eval(test_cases: list, mode: str = "precision", limit: int = 0) -> list:
    """
    批量评估。
    mode='precision': 使用精准模式 system prompt（约束+自检）
    mode='fast': 使用快速模式 system prompt（直接生成）
    """
    if limit > 0:
        test_cases = test_cases[:limit]

    if mode == "precision":
        system_prompt = (
            "你是一个严格基于事实的AI助手。回答问题时:\n"
            "1. 只陈述确定的事实，不编造\n"
            "2. 如果不确定，明确说明\n"
            "3. 如果问题有陷阱（假设了不存在的事实），请指出\n"
            "4. 用中文回答"
        )
    else:
        system_prompt = "你是一个有帮助的AI助手，请简洁回答。"

    evaluator = HallucinationEvaluator()
    results = []
    total = len(test_cases)

    print(f"\n开始评估 {total} 个用例 ({mode} 模式)...\n")

    for i, tc in enumerate(test_cases):
        try:
            answer, latency = qwen_ask(
                system_prompt, tc.question,
                max_tokens=800 if mode == "precision" else 400,
            )
        except Exception as e:
            answer = f"[API错误] {e}"
            latency = 0

        eval_result = evaluator.evaluate_answer(tc, answer, latency)
        results.append(eval_result)

        status = "[PASS]" if eval_result.passed else "[FAIL]"
        print(f"  [{i+1}/{total}] {status} | {tc.category} | "
              f"score={eval_result.score:.2f} | {tc.question[:50]}...")

        if (i + 1) % 20 == 0:
            summary = evaluator.evaluate_batch(results)
            print(f"  --- 中期统计: 准确率 {summary['accuracy']:.1%} ---")

    return results


def main():
    print("=" * 60)
    print("  千问幻觉缓解 - 批量评估")
    print(f"  模型: {MODEL}")
    print("=" * 60)

    # ── 1. 内置测试集 ──
    print("\n[加载] 内置中文幻觉测试集")
    builtin = get_all_builtin_tests()
    print(f"  内置用例: {len(builtin)} 条 (8个类别)")

    # 类别统计
    from collections import Counter
    cats = Counter(tc.category for tc in builtin)
    for cat, count in cats.items():
        from utils.test_datasets import BUILTIN_HALLUCINATION_TESTS
        label = BUILTIN_HALLUCINATION_TESTS.get(cat, {}).get("label", cat)
        print(f"    {cat} ({label}): {count} 条")

    # ── 2. 合成对抗用例 ──
    print("\n[生成] 合成对抗测试用例")
    synthetic = DatasetDownloader.create_synthetic_hard_cases(20)
    print(f"  合成用例: {len(synthetic)} 条")

    # ── 3. 执行评估 ──
    all_cases = builtin + synthetic
    print(f"\n总计: {len(all_cases)} 个测试用例")

    # 精准模式评估
    results = run_eval(all_cases, mode="precision")

    # ── 4. 输出报告 ──
    evaluator = HallucinationEvaluator()
    summary = evaluator.evaluate_batch(results)
    evaluator.print_report(summary)

    # ── 5. 保存详细结果 ──
    output_path = "/tmp/hallucination_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": summary["total"],
                "passed": summary["passed"],
                "accuracy": summary["accuracy"],
                "avg_score": summary["avg_score"],
                "avg_latency_ms": summary["avg_latency_ms"],
                "hallucination_rate": summary["hallucination_rate"],
                "per_category": summary["per_category"],
            },
            "details": [
                {
                    "id": r.test_case.id,
                    "question": r.test_case.question,
                    "expected": r.test_case.expected_answer[:200],
                    "answer": r.model_answer[:500],
                    "score": r.score,
                    "passed": r.passed,
                    "category": r.test_case.category,
                    "latency_ms": r.latency_ms,
                    "notes": r.notes,
                }
                for r in results
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {output_path}")

    # ── 6. 失败用例分析 ──
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n{'=' * 55}")
        print(f"  失败用例分析 ({len(failed)} 条)")
        print(f"{'=' * 55}")
        for r in failed[:10]:
            print(f"  [{r.test_case.category}] {r.test_case.question}")
            print(f"  回答: {r.model_answer[:200]}...")
            print(f"  得分: {r.score:.2f} | {r.notes}")
            print()

    print("\n评估完成!")


if __name__ == "__main__":
    main()
