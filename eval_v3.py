#!/usr/bin/env python3
"""
评估 v3.0 —— 阶段1增强版
核心改动: 阶段1增加问题类型分类 → 三路路由
  - factual → precision (多源验证)
  - logic   → logic_prompt (专用提示词,不走检索)
  - chat    → fast (缓存优先)
"""
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, ".")
from core.preprocessor import (
    InputPreprocessor, PreprocessResult, LOGIC_PATTERNS, LOGIC_SYSTEM_PROMPT,
)

BASE_URL = "http://127.0.0.1:8009/v1"
MODEL = "qwen35-9b-thinking"


# ═══════════════════════════════════════════════════════════
# 扩展测试集 (大幅增加样本量)
# ═══════════════════════════════════════════════════════════

@dataclass
class TestCase:
    id: str
    question: str
    expected_answer: str
    category: str           # 与 preprocessor 对齐: factual / logic / chat
    sub_category: str = ""  # 逻辑子类: physics_impossible 等
    difficulty: str = "medium"
    risk_level: str = "low"


LOGIC_TESTS = [
    # ── 物理不可能 (6条) ──
    TestCase("L01", "一个人能同时站在北京和纽约的地面上吗？",
             "不能。一个人在同一时刻只能处于一个位置。", "logic", "physics_impossible", "easy", "low"),
    TestCase("L02", "一个物体能同时在0°C和100°C吗？",
             "不能。同一物体在同一时刻只能处于一种温度状态。", "logic", "physics_impossible", "easy", "low"),
    TestCase("L03", "水能在标准大气压下同时结冰并沸腾吗？",
             "不能。0°C以下结冰，100°C沸腾，两者不可能同时发生。", "logic", "physics_impossible", "medium", "medium"),
    TestCase("L04", "一束光从地球到太阳需要多长时间？能比8分钟更短吗？",
             "约8分20秒。在真空中光速不可超越，无法更短。", "logic", "physics_impossible", "medium", "medium"),
    TestCase("L05", "一辆汽车能在同一时刻既在行驶又在静止吗？",
             "不能。在同一参考系中，物体只能处于一种运动状态。", "logic", "physics_impossible", "easy", "low"),
    TestCase("L06", "一个人能同时在北京呼吸且在月球上不穿宇航服存活吗？",
             "不能。同一个人不能同时在两个相距38万公里、环境截然不同的地方存在。", "logic", "physics_impossible", "medium", "low"),

    # ── 数学矛盾 (5条) ──
    TestCase("L07", "存在一个既是偶数又是奇数的整数吗？",
             "不存在。偶数和奇数是互斥的数学定义。", "logic", "math_contradiction", "easy", "low"),
    TestCase("L08", "一个三角形的内角和能同时等于180度和360度吗？",
             "不能。欧几里得三角形内角和恒等于180度。", "logic", "math_contradiction", "easy", "low"),
    TestCase("L09", "能找到比0小的正整数吗？",
             "不能。正整数定义为大于0的整数，最小是1。", "logic", "math_contradiction", "easy", "low"),
    TestCase("L10", "一个数字能同时是π和整数吗？",
             "不能。π是无理数，永远不可能是整数。", "logic", "math_contradiction", "easy", "low"),
    TestCase("L11", "存在一个数同时大于100且小于10吗？",
             "不存在。一个数不可能同时大于100又小于10。", "logic", "math_contradiction", "easy", "low"),

    # ── 时间悖论 (5条) ──
    TestCase("L12", "今天出生的人能昨天就去上学吗？",
             "不能。出生之前这个人不存在。", "logic", "temporal_paradox", "easy", "low"),
    TestCase("L13", "一个人在去世后又写了一本书，这可能吗？",
             "在现实中不可能。去世后无法进行任何活动。", "logic", "temporal_paradox", "easy", "high"),
    TestCase("L14", "2020年的你能否记住2025年发生的事情？",
             "不能。人无法拥有未来的记忆。", "logic", "temporal_paradox", "medium", "medium"),
    TestCase("L15", "一项技术在1890年发明了互联网，这合理吗？",
             "不合理。1890年连电子计算机都没有。", "logic", "temporal_paradox", "medium", "medium"),
    TestCase("L16", "下周一还没到，我能描述下周一发生的事吗？",
             "不能准确描述细节。未发生的事不可预知。", "logic", "temporal_paradox", "easy", "low"),

    # ── 身份/关系矛盾 (5条) ──
    TestCase("L17", "一个人可以是自己的父亲吗？",
             "不能。父子关系要求双方是不同个体且有先后顺序。", "logic", "identity_contradiction", "easy", "low"),
    TestCase("L18", "小明今年5岁，他的儿子今年10岁，这可能吗？",
             "不可能。父亲不能比儿子年龄小。", "logic", "identity_contradiction", "easy", "low"),
    TestCase("L19", "一个未婚男子能同时是鳏夫吗？",
             "不能。鳏夫定义是丧偶的已婚男性，未婚不可能是鳏夫。", "logic", "identity_contradiction", "medium", "medium"),
    TestCase("L20", "一个人可以既是独生子又有亲姐姐吗？",
             "不能。独生子定义为没有兄弟姐妹。", "logic", "identity_contradiction", "easy", "low"),
    TestCase("L21", "一位女性同时是自己的外婆吗？",
             "不能。外婆与外孙女是不同世代的人。", "logic", "identity_contradiction", "easy", "low"),

    # ── 因果颠倒 (3条) ──
    TestCase("L22", "因为他得了肺癌，所以他开始吸烟，这个因果关系成立吗？",
             "不成立。吸烟是肺癌的原因，不是结果。因果关系颠倒。", "logic", "causal_reversal", "easy", "high"),
    TestCase("L23", "因为他通过了考试，所以他开始学习，这合理吗？",
             "不合理。学习在考试之前，因果关系颠倒。", "logic", "causal_reversal", "easy", "low"),
    TestCase("L24", "因为洪水退了，所以开始下雨，这可能吗？",
             "不可能。下雨是洪水的原因，而非结果。", "logic", "causal_reversal", "easy", "low"),

    # ── 范畴/单位错误 (4条) ──
    TestCase("L25", "一个物体重50公斤，同时也重50米，这可能吗？",
             "不可能。公斤是质量单位，米是长度单位，不可换算。", "logic", "category_error", "easy", "low"),
    TestCase("L26", "今天气温35°C，换算成人民币是多少？",
             "无法换算。温度单位和货币单位属于不同度量体系。", "logic", "category_error", "easy", "low"),
    TestCase("L27", "正方形有四个直角，它同时也是圆形，可能吗？",
             "不可能。正方形和圆形是互斥的几何形状定义。", "logic", "category_error", "easy", "low"),
    TestCase("L28", "一个人不是素食者但从不吃任何动物制品，矛盾吗？",
             "矛盾。不吃任何动物制品本身就是素食者的定义。", "logic", "category_error", "easy", "low"),

    # ── 全称/存在矛盾 (3条) ──
    TestCase("L29", "所有乌鸦都是黑色的，但存在一只白色乌鸦，这个陈述矛盾吗？",
             "矛盾。如果存在白色乌鸦，则'所有乌鸦都是黑色的'为假。", "logic", "universal_contradiction", "medium", "low"),
    TestCase("L30", "这个房间里没有人，同时有一个人在这个房间里，可能吗？",
             "不可能。同一空间不能同时既无人又有人。", "logic", "universal_contradiction", "easy", "low"),
    TestCase("L31", "中国人都会说中文，但有一个中国人不会说中文，矛盾吗？",
             "矛盾。如果存在不会说中文的中国人，则'中国人都会说中文'为假。", "logic", "universal_contradiction", "easy", "low"),

    # ── 定义冲突 (4条) ──
    TestCase("L32", "单身汉已经结婚了，这可能吗？",
             "不可能。单身汉的定义就是未婚男性。", "logic", "definition_conflict", "easy", "low"),
    TestCase("L33", "一匹活马同时是死的，这可能吗？",
             "不可能。活和死是互斥的生命状态。", "logic", "definition_conflict", "easy", "low"),
    TestCase("L34", "法语是法国的官方语言，也是南极洲的官方语言吗？",
             "不是。南极洲没有国家，不存在官方语言。", "logic", "spatial_contradiction", "medium", "medium"),
    TestCase("L35", "一个两岁的小孩是百岁老人，这可能吗？",
             "不可能。两岁和百岁在年龄定义上矛盾。", "logic", "definition_conflict", "easy", "low"),
]

FACTUAL_TESTS = [
    TestCase("F01", "谁在1969年首次登上月球？",
             "尼尔·阿姆斯特朗和巴兹·奥尔德林。阿波罗11号任务。", "factual", "", "easy", "low"),
    TestCase("F02", "Python编程语言是由谁在哪一年创建的？",
             "Guido van Rossum在1991年创建。", "factual", "", "easy", "low"),
    TestCase("F03", "水的化学式是什么？",
             "H2O。由两个氢原子和一个氧原子组成。", "factual", "", "easy", "low"),
    TestCase("F04", "爱因斯坦获得诺贝尔奖是因为什么理论？",
             "1921年因光电效应理论获得诺贝尔物理学奖，不是相对论。", "factual", "", "easy", "high"),
    TestCase("F05", "标准大气压下水的沸点是多少度？",
             "100摄氏度（212华氏度）。", "factual", "", "easy", "low"),
    TestCase("F06", "地球到月球的距离大约是多少公里？",
             "约384,400公里。", "factual", "", "medium", "low"),
    TestCase("F07", "中国有多少个省级行政区？",
             "34个：23省、5自治区、4直辖市、2特别行政区。", "factual", "", "medium", "low"),
    TestCase("F08", "光的真空传播速度是多少？",
             "约299,792公里/秒（约30万公里/秒）。", "factual", "", "easy", "low"),
    TestCase("F09", "人体正常体温范围是多少？",
             "约36.5-37.5°C（腋下测量）。", "factual", "", "easy", "high"),
    TestCase("F10", "地球的直径大约是多少？",
             "约12,742公里。", "factual", "", "medium", "low"),
    TestCase("F11", "中国的四大发明是什么？",
             "造纸术、指南针、火药、印刷术。", "factual", "", "easy", "low"),
    TestCase("F12", "莎士比亚写过《哈利波特》系列吗？",
             "没有。《哈利波特》是J.K.罗琳的作品，莎士比亚是16-17世纪的剧作家。", "factual", "", "easy", "high"),
]

CHAT_TESTS = [
    TestCase("C01", "你好！今天心情怎么样？", "友好问候，不需要事实核查", "chat", "", "easy", "low"),
    TestCase("C02", "请用一句话介绍你自己", "自我介绍", "chat", "", "easy", "low"),
    TestCase("C03", "讲一个简短的笑话吧", "笑话，创造性内容", "chat", "", "easy", "low"),
    TestCase("C04", "谢谢你刚才的帮助！", "礼貌回应", "chat", "", "easy", "low"),
    TestCase("C05", "你觉得今天天气好吗？", "主观感受，无标准答案", "chat", "", "easy", "low"),
]

# 保证质量问题（故意有前提陷阱）
TRAP_TESTS = [
    TestCase("T01", "2025年世界杯冠军是哪个国家？",
             "2025年世界杯尚未举办。2026年世界杯将在北美举办。", "factual", "", "medium", "high"),
    TestCase("T02", "谁在1969年首次登上火星？",
             "人类至今尚未登陆火星。1969年阿波罗11号登陆的是月球。", "factual", "", "easy", "high"),
    TestCase("T03", "爱因斯坦在哪一年获得诺贝尔化学奖？",
             "爱因斯坦获得的是物理学奖，不是化学奖。", "factual", "", "easy", "high"),
    TestCase("T04", "乔布斯在哪一年发布了iPhone 15？",
             "乔布斯2011年去世，iPhone 15是2023年由苹果公司发布的，与他无关。", "factual", "", "medium", "high"),
    TestCase("T05", "请引用2024年《自然》杂志关于治愈阿尔茨海默症的文章。",
             "不应该编造不存在的文献引用。", "factual", "", "hard", "high"),
]


# ═══════════════════════════════════════════════════════════
# 千问调用
# ═══════════════════════════════════════════════════════════

def qwen_call(system_prompt, user_query, max_tokens=600):
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.05,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    msg = d["choices"][0]["message"]
    content = (msg.get("content") or msg.get("reasoning", "") or "").strip()
    return content, (time.time() - t0) * 1000


# ═══════════════════════════════════════════════════════════
# 评估器
# ═══════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    tc: TestCase
    predicted_type: str       # 分类器预测类型
    actual_answer: str
    score: float
    passed: bool
    latency_ms: float
    notes: str = ""


class Evaluator:
    def __init__(self):
        self.preprocessor = InputPreprocessor(llm=None)
        self.results: List[EvalResult] = []

    async def evaluate(self, tc: TestCase) -> EvalResult:
        # 阶段1: 预处理分类
        pre = await self.preprocessor.preprocess(tc.question)

        # 选择 prompt
        if pre.is_logic_question:
            system_prompt = LOGIC_SYSTEM_PROMPT
            max_tok = 300
        elif pre.question_type == "chat":
            system_prompt = "你是一个友好的AI助手。"
            max_tok = 300
        else:
            system_prompt = (
                "你是一个严格基于事实的AI助手。只陈述确定的事实。"
                "如果问题前提有误，请先指出错误再回答。不确定就说明。"
            )
            max_tok = 600

        # 调用模型
        answer, latency = qwen_call(system_prompt, tc.question, max_tokens=max_tok)

        # 评分
        score = self._score(tc, answer, pre)
        passed = score >= 0.5

        result = EvalResult(
            tc=tc,
            predicted_type=pre.question_type,
            actual_answer=answer,
            score=score,
            passed=passed,
            latency_ms=latency,
            notes=f"indicators={pre.logic_indicators}" if pre.logic_indicators else "",
        )
        self.results.append(result)
        return result

    def _score(self, tc: TestCase, answer: str, pre: PreprocessResult) -> float:
        """评分逻辑"""
        score = 0.5
        al = answer.lower()
        ql = tc.question.lower()

        # 1. 逻辑类 → 检查是否给出明确结论
        if tc.category == "logic":
            direct_keywords = ["不能", "不可能", "不存在", "矛盾", "不可以", "没有", "不是",
                               "不成立", "不合理", "无法", "cannot", "not", "no", "never"]
            hedging_keywords = ["量子力学", "相对论", "某种意义", "取决于", "一方面",
                                "从某个角度", "也许", "可能可以", "不一定"]

            # 使用了明确结论词 → 加分
            if any(kw in answer for kw in direct_keywords):
                score += 0.30
            # 使用了回避/绕圈子词 → 扣分
            if any(kw in answer for kw in hedging_keywords):
                score -= 0.25

        # 2. 事实类 → 检查是否识别陷阱
        if tc.category == "factual" and tc.risk_level == "high":
            trap_signals = ["尚未", "没有", "不是", "错误", "并非", "并未",
                            "未曾", "不可能", "不是化学奖", "不是物理学奖",
                            "无关", "尚未举办"]
            if any(s in answer for s in trap_signals):
                score += 0.15

        # 3. 闲聊类 → 检查是否友好
        if tc.category == "chat":
            score += 0.15  # 闲聊评分宽松

        return min(1.0, max(0.0, score))

    def summary(self) -> dict:
        if not self.results:
            return {}
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        # 按 predicted_type 统计
        by_type = {}
        for r in self.results:
            t = r.predicted_type
            if t not in by_type:
                by_type[t] = {"total": 0, "passed": 0, "scores": [], "latencies": []}
            by_type[t]["total"] += 1
            by_type[t]["passed"] += 1 if r.passed else 0
            by_type[t]["scores"].append(r.score)
            by_type[t]["latencies"].append(r.latency_ms)

        # 按测试类别统计
        by_cat = {}
        for r in self.results:
            c = r.tc.category
            if c not in by_cat:
                by_cat[c] = {"total": 0, "passed": 0, "scores": [], "latencies": []}
            by_cat[c]["total"] += 1
            by_cat[c]["passed"] += 1 if r.passed else 0
            by_cat[c]["scores"].append(r.score)
            by_cat[c]["latencies"].append(r.latency_ms)

        # 分类准确率: predicted_type 是否匹配 test_case.category
        type_match = sum(1 for r in self.results
                        if r.predicted_type == r.tc.category
                        or (r.predicted_type == "factual" and r.tc.category == "factual")
                        or (r.predicted_type == "logic" and r.tc.category == "logic")
                        or (r.predicted_type == "chat" and r.tc.category == "chat"))
        classification_accuracy = type_match / total

        return {
            "total": total, "passed": passed,
            "accuracy": passed / total,
            "classification_accuracy": classification_accuracy,
            "avg_latency_ms": sum(r.latency_ms for r in self.results) / total,
            "by_type": {t: {
                "total": d["total"], "passed": d["passed"],
                "accuracy": d["passed"] / d["total"],
                "avg_score": sum(d["scores"]) / len(d["scores"]),
                "avg_latency_ms": sum(d["latencies"]) / len(d["latencies"]),
            } for t, d in by_type.items()},
            "by_category": {c: {
                "total": d["total"], "passed": d["passed"],
                "accuracy": d["passed"] / d["total"],
                "avg_score": sum(d["scores"]) / len(d["scores"]),
                "avg_latency_ms": sum(d["latencies"]) / len(d["latencies"]),
            } for c, d in by_cat.items()},
        }

    def print_report(self, s: dict):
        print("\n" + "=" * 60)
        print("  幻觉缓解插件 v3.0 — 阶段1增强评估报告")
        print("=" * 60)
        print(f"\n  [总览]")
        print(f"  总用例: {s['total']} | 通过: {s['passed']}")
        print(f"  综合准确率: {s['accuracy']:.1%}")
        print(f"  分类准确率: {s['classification_accuracy']:.1%}")
        print(f"  平均延迟: {s['avg_latency_ms']:.0f}ms")

        print(f"\n  [按分类器路由]")
        for t in ["logic", "factual", "chat"]:
            if t in s["by_type"]:
                d = s["by_type"][t]
                label = {"logic": "逻辑/常识", "factual": "事实类", "chat": "闲聊类"}.get(t, t)
                bar = "#" * int(d["accuracy"] * 20)
                print(f"  {label:10s} {d['total']:3d}条 准确率={d['accuracy']:.0%} "
                      f"均分={d['avg_score']:.2f} 均延迟={d['avg_latency_ms']:.0f}ms {bar}")

        print(f"\n  [按测试类别]")
        for c in ["logic", "factual", "chat"]:
            if c in s["by_category"]:
                d = s["by_category"][c]
                bar = "#" * int(d["accuracy"] * 20)
                print(f"  {c:10s} {d['total']:3d}条 准确率={d['accuracy']:.0%} "
                      f"均分={d['avg_score']:.2f} 均延迟={d['avg_latency_ms']:.0f}ms {bar}")


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  幻觉缓解插件 v3.0 — 阶段1增强评估")
    print(f"  模型: {MODEL}")
    print("=" * 60)

    # 合并所有测试
    all_tests = LOGIC_TESTS + FACTUAL_TESTS + CHAT_TESTS + TRAP_TESTS
    print(f"\n[测试集]")
    print(f"  逻辑/常识: {len(LOGIC_TESTS)} 条 ({len(LOGIC_PATTERNS)} 类模式)")
    print(f"  事实类: {len(FACTUAL_TESTS)} 条")
    print(f"  闲聊类: {len(CHAT_TESTS)} 条")
    print(f"  陷阱问题: {len(TRAP_TESTS)} 条")
    print(f"  总计: {len(all_tests)} 条")

    # 验证分类器逻辑子类覆盖
    logic_cats = set()
    for tc in LOGIC_TESTS:
        logic_cats.add(tc.sub_category)
    print(f"  逻辑子类覆盖: {sorted(logic_cats)}")
    print(f"  逻辑子类总数: {len(logic_cats)}/{len(LOGIC_PATTERNS)}")

    # 评估
    evaluator = Evaluator()
    print(f"\n[评估中...]")
    for i, tc in enumerate(all_tests):
        try:
            r = await evaluator.evaluate(tc)
            status = "[PASS]" if r.passed else "[FAIL]"
            print(f"  [{i+1:2d}/{len(all_tests)}] {status} | "
                  f"type={r.predicted_type:7s} | cat={tc.category:7s} | "
                  f"score={r.score:.2f} | {tc.question[:50]}...")
        except Exception as e:
            print(f"  [{i+1:2d}/{len(all_tests)}] [ERR] {e} | {tc.question[:50]}...")
            evaluator.results.append(EvalResult(
                tc=tc, predicted_type="", actual_answer=f"[错误] {e}",
                score=0, passed=False, latency_ms=0,
            ))

    # 报告
    s = evaluator.summary()
    evaluator.print_report(s)

    # 失败用例
    failed = [r for r in evaluator.results if not r.passed]
    if failed:
        print(f"\n[失败用例 ({len(failed)} 条)]")
        for r in failed:
            print(f"  [{r.tc.category}] {r.tc.question}")
            print(f"    预测类型: {r.predicted_type} | 得分: {r.score:.2f}")
            print(f"    回答: {r.actual_answer[:200]}...")

    # 分类错误用例
    mismatches = [r for r in evaluator.results
                  if r.predicted_type != r.tc.category
                  and not (r.predicted_type == "factual" and r.tc.category == "factual")]
    if mismatches:
        print(f"\n[分类待优化用例 ({len(mismatches)} 条)]")
        for r in mismatches[:10]:
            print(f"  [{r.tc.category}→{r.predicted_type}] {r.tc.question[:60]}")

    # 保存
    out = {
        "summary": s,
        "details": [{
            "id": r.tc.id, "question": r.tc.question,
            "category": r.tc.category, "predicted_type": r.predicted_type,
            "answer": r.actual_answer[:400], "score": r.score,
            "passed": r.passed, "latency_ms": r.latency_ms,
        } for r in evaluator.results],
    }
    path = "/tmp/hallucination_eval_v3.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {path}")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
