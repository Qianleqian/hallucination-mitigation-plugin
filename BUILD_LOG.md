# 构建日志 - 大模型幻觉缓解通用插件 (千问版)

## 构建时间
2026-07-16 21:48 CST

## 项目概述
基于《面向大模型幻觉缓解的通用插件架构与闭环流程完整方案》文档，
采用「在线双路径推理 + 离线持续学习」架构，以通义千问 (Qwen) 为插件目标模型。

## 文件清单 (共 22 个源文件)

### 配置层
- `config/__init__.py` - 配置模块
- `config/settings.py` - 全局配置中心（阶段0：插件初始化与配置）

### 核心引擎层
- `core/__init__.py`
- `core/plugin_base.py` - 插件基类（MCP协议+ToolRegistry）
- `core/preprocessor.py` - 输入预处理器（阶段1）
- `core/cache_manager.py` - 向量缓存管理（阶段2.1）
- `core/search_aggregator.py` - 多源搜索聚合（阶段2.2）
- `core/fact_verifier.py` - 事实验证（阶段2.5）
- `core/dual_mode_engine.py` - 双模式推理引擎（阶段2核心）
- `core/offline_learner.py` - 离线学习闭环（阶段3）
- `core/feedback_handler.py` - 用户交互增强（阶段4）

### 模型适配层
- `models/__init__.py`
- `models/llm_base.py` - LLM 统一基类接口
- `models/qwen_adapter.py` - 通义千问适配器（DashScope API）
- `models/tool_registry.py` - 工具注册中心

### 工具层
- `utils/__init__.py`
- `utils/text_utils.py` - 文本处理工具
- `utils/metrics.py` - 评估指标收集器

### 入口
- `main.py` - 主入口（命令行交互 / 单次提问 / 离线学习 / WebSocket）
- `demo.py` - 功能演示脚本（7个演示场景）

### 配置
- `requirements.txt` - Python 依赖清单

## 测试结果

### Demo 1: 预处理与意图识别 [OK]
- 5 个测试查询全部通过
- 高风险检测正确（投资股票 -> 精准模式）
- 意图分类正确

### Demo 2: 缓存管理 [OK]
- 存储/检索正常
- 关键词匹配回退方案可用
- 过期淘汰机制正常

### Demo 3: 多源搜索聚合 [OK]
- DuckDuckGo 集成正常
- 超时控制正常
- 优雅降级正常

### Demo 4: 事实验证 [OK]
- NLI 模型加载失败时回退到启发式
- 来源优先级排序正确
- 冲突信息处理正确

### Demo 5: 离线学习 [OK]
- 样本不足时自动跳过
- 数据清洗/数据集构建正常
- 日志记录正常

### Demo 6: 用户反馈 [OK]
- 反馈记录/统计正常
- 好评样本提取正常

### Demo 7: 完整流程 [SKIP]
- 正确检测 DASHSCOPE_API_KEY 缺失
- 给出清晰配置提示

## 环境依赖
- Python 3.10+
- dashscope (千问 API)
- sentence-transformers (可选，语义匹配)
- duckduckgo-search (可选，免费搜索)
- transformers (可选，NLI 验证)
- chromadb (可选，向量数据库)
- websockets (可选，实时推送)

## 下一步
1. 设置 DASHSCOPE_API_KEY 环境变量以启用千问
2. 安装可选依赖以获得完整功能体验
3. 生产部署：添加持久化向量数据库、日志系统、监控告警
