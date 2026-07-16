# 面向大模型幻觉缓解的通用插件架构

## 千问版 (Qwen Edition)

基于《面向大模型幻觉缓解的通用插件架构与闭环流程完整方案》文档实现，
采用 **「在线双路径推理 + 离线持续学习」** 架构，以 **通义千问 (Qwen)** 为插件目标模型。

---

## 架构总览

```
                         用户输入
                            |
                     +------v-------+
                     |  阶段1: 预处理  |
                     | 清洗/意图/不确定性|
                     +------+-------+
                            |
                   +--------v---------+
                   |   模式智能决策     |
                   | 快速 / 精准 / 自动 |
                   +---+----------+---+
                       |          |
              +--------v---+  +--v-----------+
              |  阶段2.1   |  |  阶段2.2      |
              | 快速模式    |  | 精准模式       |
              | 缓存->生成  |  | 搜索->验证->生成|
              +------------+  +---------------+
                       |          |
                       +----+-----+
                            |
                     +------v-------+
                     |  阶段4: 输出   |
                     | 流式进度/反馈  |
                     +--------------+

              离线 (异步，不阻塞在线推理):
              +---------------------------+
              |  阶段3: 离线学习闭环        |
              |  数据采集->清洗->微调->部署  |
              +---------------------------+
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆或进入项目目录
cd hallucination-mitigation-plugin

# 安装核心依赖
pip install dashscope>=1.20.0

# 安装可选依赖（推荐）
pip install duckduckgo-search sentence-transformers transformers

# 安装全部依赖
pip install -r requirements.txt
```

### 2. 设置千问 API Key

```bash
# Linux/Mac
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# Windows PowerShell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"

# Windows CMD
set DASHSCOPE_API_KEY=your-dashscope-api-key
```

> 获取 API Key: [https://dashscope.console.aliyun.com/](https://dashscope.console.aliyun.com/)

### 3. 运行演示

```bash
# 运行 7 个功能演示（无需 API Key 也能运行大部分）
python demo.py
```

### 4. 交互式问答

```bash
# 启动交互模式
python main.py

# 指定模型版本
python main.py --model qwen-max
python main.py --model qwen2.5-72b-instruct
```

---

## 使用方式

### 交互模式命令

```
[auto] 请输入问题: 什么是量子计算？
[auto] 请输入问题: /fast          # 切换至快速模式
[fast] 请输入问题: /precision     # 切换至精准模式
[precision] 请输入问题: /stats    # 查看会话统计
[precision] 请输入问题: /feedback # 对上次回答评分
[precision] 请输入问题: /exit     # 退出
```

### 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--mode` | 工作模式 (fast/precision/auto) | `--mode precision` |
| `--query` | 单次提问 | `--query "什么是量子计算？"` |
| `--model` | 千问模型版本 | `--model qwen-max` |
| `--offline-learn` | 执行离线学习周期 | `--offline-learn` |
| `--serve` | 启动 WebSocket 服务 | `--serve` |
| `--stats` | 显示会话统计 | `--stats` |

### 单次提问示例

```bash
# 精准模式（多源搜索 + 交叉验证）
python main.py --mode precision --query "2024年诺贝尔奖得主是谁？"

# 快速模式（缓存优先）
python main.py --mode fast --query "什么是机器学习？"

# 自动模式（高风险自动切精准）
python main.py --query "感冒了应该吃什么药？"
```

---

## 两种工作模式

### 快速模式 (Fast Mode)

**适用场景**: 日常闲聊、简单常识、低风险资讯

**流程**:
1. 查询本地缓存（阈值 0.70）
2. 命中 -> 直接返回缓存答案
3. 未命中 -> LLM 直接生成 + 免责声明

**性能**: 响应延迟 < 200ms（缓存命中时）

### 精准模式 (Precision Mode)

**适用场景**: 专业咨询、知识核查、学术查询、高风险领域

**流程**:
1. 查询缓存（阈值 0.85）
2. 查询改写（LLM 生成多维度检索指令）
3. 多源并行搜索（DuckDuckGo + Bing + SerpAPI）
4. 交叉验证（NLI 模型 + 加权投票）
5. 提取结构化事实
6. 基于验证事实生成答案（强制约束）
7. 更新本地缓存

**自动触发条件**:
- 查询包含金融/医疗/法律/药品等高风险关键词
- 模型不确定性 > 0.65

---

## 千问模型支持

| 模型 | 说明 | 推荐场景 |
|------|------|---------|
| `qwen-turbo` | 速度最快 | 快速模式 |
| `qwen-plus` | 平衡性能 | 通用场景（默认） |
| `qwen-max` | 最强能力 | 精准模式 |
| `qwen2.5-72b-instruct` | 开源旗舰 | 本地部署 |
| `qwen-long` | 长文本 | 文档分析 |

```python
# API 调用示例
from models.llm_base import LLMConfig
from models.qwen_adapter import QwenAdapter

qwen = QwenAdapter(LLMConfig(
    model="qwen-max",
    api_key="your-api-key",
    temperature=0.1,
))

resp = await qwen.chat([
    {"role": "user", "content": "你好"}
])
print(resp.content)
```

---

## 核心模块说明

| 模块 | 文件 | 文档对应 |
|------|------|----------|
| 插件接入 | `models/tool_registry.py` | 阶段0: MCP协议+ToolRegistry |
| 千问适配 | `models/qwen_adapter.py` | 阶段0: 模型无关化接入 |
| 预处理 | `core/preprocessor.py` | 阶段1: 清洗/意图/不确定性 |
| 缓存 | `core/cache_manager.py` | 阶段2.1: 向量缓存/TTL淘汰 |
| 搜索 | `core/search_aggregator.py` | 阶段2.2: 多源并行/查询改写 |
| 验证 | `core/fact_verifier.py` | 阶段2.5: NLI校对/冲突排序 |
| 引擎 | `core/dual_mode_engine.py` | 阶段2: 双模式路由+推理 |
| 学习 | `core/offline_learner.py` | 阶段3: 数据采集/微调/评估 |
| 交互 | `core/feedback_handler.py` | 阶段4: WebSocket/反馈/切换 |

---

## 离线学习闭环

系统自动执行以下周期（默认每周）：

```
高置信事实采集(>0.9) -> 数据清洗去重 -> SFT/DPO数据集 ->
LoRA微调 -> TIES/SLERP权重融合(30%新+70%旧) ->
模型评估 -> 通过则部署/失败则回滚 -> 缓存同步清理
```

手动触发:
```bash
python main.py --offline-learn
```

程序化触发:
```python
from core.offline_learner import OfflineLearner
from core.cache_manager import CacheManager

cache = CacheManager()
learner = OfflineLearner(cache)
report = await learner.run_cycle()
```

---

## 扩展新模型

基于 MCP 协议理念，添加新模型仅需实现 `BaseLLM` 接口:

```python
from models.llm_base import BaseLLM, LLMConfig, LLMResponse

class MyModelAdapter(BaseLLM):
    def __init__(self, config: LLMConfig):
        super().__init__(config)

    async def chat(self, messages, tools=None, **kwargs) -> LLMResponse:
        # 实现你的模型调用逻辑
        ...

    async def chat_stream(self, messages, **kwargs):
        # 实现流式调用
        ...

    async def get_logits_uncertainty(self, messages) -> float:
        # 实现不确定性估计
        ...
```

---

## 配置参数

编辑 `config/settings.py` 或运行时修改:

```python
from config.settings import config

# 缓存阈值
config.fast_mode_confidence_threshold = 0.70
config.precision_mode_confidence_threshold = 0.85

# 搜索超时
config.search_timeout_sec = 5

# 离线学习间隔（小时）
config.offline_learning_interval_hours = 168  # 每周

# 高质量事实筛选阈值
config.high_quality_fact_threshold = 0.90
```

---

## 项目结构

```
hallucination-mitigation-plugin/
├── config/
│   └── settings.py              # 全局配置
├── core/
│   ├── plugin_base.py           # 插件基类
│   ├── preprocessor.py          # 输入预处理
│   ├── cache_manager.py         # 缓存管理
│   ├── search_aggregator.py     # 搜索聚合
│   ├── fact_verifier.py         # 事实验证
│   ├── dual_mode_engine.py      # 双模式引擎
│   ├── offline_learner.py       # 离线学习
│   └── feedback_handler.py      # 交互增强
├── models/
│   ├── llm_base.py              # LLM 基类
│   ├── qwen_adapter.py          # 千问适配器
│   └── tool_registry.py         # 工具注册中心
├── utils/
│   ├── text_utils.py            # 文本工具
│   └── metrics.py               # 评估指标
├── data/                        # 运行时数据（自动生成）
├── main.py                      # 主入口
├── demo.py                      # 演示脚本
├── requirements.txt             # 依赖清单
├── BUILD_LOG.md                 # 构建日志
└── README.md                    # 本文件
```

---

## 预期效果

| 指标 | 快速模式 | 精准模式 |
|------|---------|---------|
| 响应延迟 | < 200ms (缓存命中) | 2-5 秒 |
| 事实准确率 | ~80% | > 90% |
| 缓存命中率 | 70-85% | 30-50% |
| 幻觉率 | 中等 | 低 |

---

## 已知限制

1. 聚焦事实性幻觉，不解决深层逻辑推理错误
2. 离线学习存在固定周期（数小时至数天），不满足实时知识更新
3. 精准模式依赖外部搜索 API 稳定性
4. NLI 模型加载需要 GPU 推荐（CPU 也可运行但较慢）

---

## License

MIT
