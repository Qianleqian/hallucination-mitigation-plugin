"""快速测试: 千问 thinking 模型的幻觉缓解效果"""
import urllib.request, json, time

BASE = "http://127.0.0.1:8009/v1"

def ask(system_prompt, user_query, max_tokens=2000, enable_thinking=False):
    payload = json.dumps({
        "model": "qwen35-9b-thinking",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }).encode()

    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())

    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    latency = (time.time() - t0) * 1000
    usage = d.get("usage", {})
    return content, reasoning, latency, usage

print("=" * 60)
print("  千问幻觉缓解 - 效果验证")
print("=" * 60)

# 测试1: 事实性问题
print("\n[测试1] 事实性问答: Python创建者")
c, r, lat, usage = ask(
    "请直接给出简洁的最终答案。",
    "Python编程语言是哪一年由谁创建的？",
    max_tokens=1000, enable_thinking=False,
)
# 过滤 thinking 标记
if "1991" in c or "1991" in r:
    print(f"  [PASS] 正确回答包含1991年")
else:
    print(f"  [CHECK] 回答: {c[:200]}")
print(f"  延迟: {lat:.0f}ms")
print(f"  Token: {usage}")

# 测试2: 幻觉检测 - 不存在的事件
print("\n[测试2] 幻觉检测: 2025世界杯冠军")
c, r, lat, usage = ask(
    "请基于事实回答。如果事件尚未发生或信息不存在，请明确说明。",
    "2025年世界杯足球赛的冠军是哪个国家？",
    max_tokens=1000, enable_thinking=False,
)
combined = (c + r).lower()
if any(w in combined for w in ["不存在", "未举办", "尚未", "还未", "没有", "does not exist", "not take place"]):
    print(f"  [PASS] 模型正确拒绝幻觉问题")
elif any(w in combined for w in ["2026", "下届"]):
    print(f"  [PASS] 模型指向2026年世界杯，未编造2025年结果")
else:
    print(f"  [CHECK] {c[:300]}")
print(f"  延迟: {lat:.0f}ms")

# 测试3: 精准事实
print("\n[测试3] 精准事实: 水的沸点")
c, r, lat, usage = ask(
    "请直接回答数字和单位。",
    "标准大气压下水的沸点是多少度？",
    max_tokens=500, enable_thinking=False,
)
if "100" in (c + r):
    print(f"  [PASS] 正确回答100度")
else:
    print(f"  [CHECK] {(c+r)[:200]}")
print(f"  延迟: {lat:.0f}ms")

# 测试4: 可能冲突的信息
print("\n[测试4] 易混淆问题: 地球到太阳距离")
c, r, lat, usage = ask(
    "请基于权威天文数据回答。",
    "地球到太阳的平均距离是多少公里？",
    max_tokens=1000, enable_thinking=False,
)
if "1.496" in (c+r) or "1.5亿" in (c+r) or "149" in (c+r):
    print(f"  [PASS] 正确回答约1.496亿公里")
else:
    print(f"  [CHECK] {(c+r)[:300]}")
print(f"  延迟: {lat:.0f}ms")

print("\n" + "=" * 60)
print("  测试完成!")
print("=" * 60)
