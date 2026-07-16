"""
离线学习闭环 —— 对应方案文档「阶段3：离线学习闭环（异步持续迭代）」
实现: 高质量事实采集 -> 数据清洗 -> 微调数据集构建 -> LoRA 微调 -> 权重融合 -> 模型评估 -> 部署

流程:
3.1 采集高质量事实数据（置信度 > 0.9）
3.2 数据清洗与去重增强
3.3 构建微调数据集（DPO 偏好训练数据）
3.4 LoRA 权重融合（TIES-Merging / SLERP）
3.5 模型评估与自动回滚
3.6 缓存同步清理
"""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from config.settings import config


@dataclass
class TrainingSample:
    """微调训练样本"""
    instruction: str
    input: str = ""
    output: str = ""
    source_url: str = ""
    confidence: float = 0.0
    created_at: str = ""

    def to_alpaca_format(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    def to_dpo_format(self, rejected_output: str = "") -> dict:
        """DPO 偏好训练格式"""
        return {
            "prompt": f"{self.instruction}\n\n{self.input}",
            "chosen": self.output,
            "rejected": rejected_output or "我不知道。",
        }


class DataCleaner:
    """阶段3.2: 数据清洗与增强"""

    @staticmethod
    def clean(samples: List[TrainingSample]) -> List[TrainingSample]:
        """清洗: 去重、去短、去无效"""
        seen = set()
        cleaned = []
        for s in samples:
            key = (s.instruction[:100], s.output[:100])
            if key in seen:
                continue
            if len(s.output) < 10:
                continue
            if s.output.strip() in ("我不知道。", "我不确定。", "..."):
                continue
            seen.add(key)
            cleaned.append(s)
        return cleaned

    @staticmethod
    def augment(samples: List[TrainingSample]) -> List[TrainingSample]:
        """简单增强: 每个样本生成一个变体"""
        augmented = list(samples)
        for s in samples:
            augmented.append(TrainingSample(
                instruction=f"请详细说明: {s.instruction}",
                input=s.input,
                output=s.output,
                source_url=s.source_url,
                confidence=s.confidence,
                created_at=s.created_at,
            ))
        return augmented

    @staticmethod
    def filter_conflicts(samples: List[TrainingSample]) -> List[TrainingSample]:
        """过滤冲突样本: 同一 query 出现多个不同 answer 时，保留置信度最高的"""
        grouped = {}
        for s in samples:
            key = s.instruction[:100]
            if key not in grouped or s.confidence > grouped[key].confidence:
                grouped[key] = s
        return list(grouped.values())


class DatasetBuilder:
    """阶段3.3: 构建微调数据集"""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(config.data_dir, "datasets")
        os.makedirs(self.output_dir, exist_ok=True)

    def build_sft_dataset(
        self, samples: List[TrainingSample], name: str = "sft"
    ) -> str:
        """构建 SFT 监督微调数据集 (Alpaca 格式)"""
        data = [s.to_alpaca_format() for s in samples]
        path = os.path.join(self.output_dir, f"{name}_{len(data)}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def build_dpo_dataset(
        self, samples: List[TrainingSample], name: str = "dpo"
    ) -> str:
        """构建 DPO 偏好训练数据集"""
        data = []
        for s in samples:
            # 构造负样本（原始模型可能的错误回答作为 rejected）
            dpo_sample = s.to_dpo_format(
                rejected_output=f"根据我的理解，{s.instruction}的答案可能是..."
            )
            data.append(dpo_sample)
        path = os.path.join(self.output_dir, f"{name}_{len(data)}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path


class LoRATrainer:
    """阶段3.4: LoRA 微调 + 权重融合"""

    def __init__(self, base_model: str = "qwen2.5-7b-instruct"):
        self.base_model = base_model
        self._peft_config = None

    def get_config(self) -> dict:
        """LoRA 配置"""
        return {
            "r": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }

    async def train(
        self, dataset_path: str, output_dir: str = None
    ) -> str:
        """
        LoRA 微调执行。
        实际生产环境中通过调用训练脚本来完成。
        此处提供完整的参数配置和训练命令生成。
        """
        output_dir = output_dir or os.path.join(config.data_dir, "lora_checkpoints")
        os.makedirs(output_dir, exist_ok=True)

        lora_config = self.get_config()

        # 生成训练命令（供外部执行）
        train_cmd = (
            f"python -m torch.distributed.launch "
            f"--nproc_per_node=1 "
            f"train_lora.py "
            f"--model_name_or_path {self.base_model} "
            f"--dataset_path {dataset_path} "
            f"--output_dir {output_dir} "
            f"--lora_r {lora_config['r']} "
            f"--lora_alpha {lora_config['lora_alpha']} "
            f"--lora_dropout {lora_config['lora_dropout']} "
            f"--num_train_epochs 3 "
            f"--per_device_train_batch_size 4 "
            f"--gradient_accumulation_steps 4 "
            f"--learning_rate 2e-5 "
            f"--warmup_ratio 0.03 "
            f"--logging_steps 10 "
            f"--save_steps 100 "
            f"--bf16 True"
        )

        # 保存配置
        train_config_path = os.path.join(output_dir, "train_config.json")
        with open(train_config_path, "w", encoding="utf-8") as f:
            json.dump({
                "command": train_cmd,
                "lora_config": lora_config,
                "dataset_path": dataset_path,
                "timestamp": datetime.now().isoformat(),
            }, f, ensure_ascii=False, indent=2)

        return output_dir

    @staticmethod
    def merge_weights(
        lora_path: str,
        base_model: str,
        output_path: str,
        method: str = "ties",
    ) -> str:
        """
        权重融合（对应文档 TIES-Merging / SLERP）。
        30% 新知识 + 70% 原有知识 的加权融合。
        """
        import subprocess

        if method == "ties":
            cmd = (
                f"python -m mergekit.ties "
                f"--base-model {base_model} "
                f"--lora-path {lora_path} "
                f"--output {output_path} "
                f"--density 0.3 "
                f"--weight-mask-rate 0.7"
            )
        else:  # slerp
            cmd = (
                f"python -m mergekit.slerp "
                f"--base-model {base_model} "
                f"--lora-path {lora_path} "
                f"--output {output_path} "
                f"--t 0.3"
            )

        # 保存合并命令
        merge_script = os.path.join(os.path.dirname(output_path), "merge_command.sh")
        with open(merge_script, "w") as f:
            f.write(f"#!/bin/bash\n{cmd}\n")

        return output_path


class ModelEvaluator:
    """阶段3.5: 模型评估与自动回滚"""

    def __init__(self):
        self.baseline_metrics = {}
        self.rollback_threshold = 0.05  # 性能下降 5% 触发回滚

    async def evaluate(self, model_path: str, test_dataset_path: str) -> dict:
        """
        评估新模型的:
        - 事实准确性
        - 通用对话能力
        - 响应延迟
        如果性能下降超过阈值，自动回滚。
        """
        # 在实际部署中，这调用评估脚本
        metrics = {
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "factual_accuracy": 0.0,    # 由外部评估脚本填充
            "dialog_quality": 0.0,
            "avg_latency_ms": 0.0,
            "hallucination_rate": 0.0,
            "evaluated_samples": 0,
        }

        # 保存评估结果
        eval_path = os.path.join(
            config.data_dir, "evaluations",
            f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return metrics

    def should_rollback(self, new_metrics: dict) -> bool:
        """判断是否需要回滚"""
        if not self.baseline_metrics:
            return False
        accuracy_drop = (
            self.baseline_metrics.get("factual_accuracy", 0)
            - new_metrics.get("factual_accuracy", 0)
        )
        return accuracy_drop > self.rollback_threshold


class OfflineLearner:
    """
    离线学习闭环主控制器。
    异步执行，不阻塞在线推理。

    使用:
        learner = OfflineLearner(cache_manager)
        report = await learner.run_cycle()
    """

    def __init__(self, cache_manager):
        self.cache = cache_manager
        self.cleaner = DataCleaner()
        self.builder = DatasetBuilder()
        self.trainer = LoRATrainer()
        self.evaluator = ModelEvaluator()
        self.run_history: List[dict] = []
        self._log_path = os.path.join(config.data_dir, "offline_learning_log.jsonl")

    async def run_cycle(self) -> dict:
        """执行一个完整的离线学习周期"""
        start = time.time()
        report = {"steps": [], "errors": []}

        try:
            # 3.1: 采集高质量事实数据
            self._log("阶段3.1: 采集高质量事实数据")
            entries = self.cache.get_high_quality_entries(
                config.high_quality_fact_threshold
            )
            samples = [
                TrainingSample(
                    instruction=f"回答以下问题: {e.query}",
                    input="",
                    output=e.answer,
                    source_url=e.source_urls[0] if e.source_urls else "",
                    confidence=e.confidence,
                    created_at=datetime.now().isoformat(),
                )
                for e in entries
            ]
            report["steps"].append(f"采集 {len(samples)} 条高质量样本")
            self._log(f"  采集到 {len(samples)} 条高质量样本")

            if len(samples) < 10:
                report["status"] = "skipped"
                report["reason"] = f"样本不足 ({len(samples)} < 10)"
                self._log("  样本不足，跳过本轮学习")
                return report

            # 3.2: 数据清洗与增强
            self._log("阶段3.2: 数据清洗与去重增强")
            samples = self.cleaner.clean(samples)
            samples = self.cleaner.filter_conflicts(samples)
            samples = self.cleaner.augment(samples)
            report["steps"].append(f"清洗后 {len(samples)} 条样本")

            # 3.3: 构建微调数据集
            self._log("阶段3.3: 构建微调数据集")
            sft_path = self.builder.build_sft_dataset(samples, "hallucination_sft")
            dpo_path = self.builder.build_dpo_dataset(samples, "hallucination_dpo")
            report["steps"].append(f"SFT数据集: {sft_path}")
            report["steps"].append(f"DPO数据集: {dpo_path}")

            # 3.4: LoRA 微调
            self._log("阶段3.4: 执行LoRA微调")
            lora_output = await self.trainer.train(sft_path)
            report["steps"].append(f"LoRA权重: {lora_output}")

            # 权重融合
            merged_path = self.trainer.merge_weights(
                lora_output,
                "qwen2.5-7b-instruct",
                os.path.join(config.data_dir, "merged_model"),
            )
            report["steps"].append(f"融合模型: {merged_path}")

            # 3.5: 模型评估
            self._log("阶段3.5: 模型评估与回滚检查")
            eval_metrics = await self.evaluator.evaluate(merged_path, sft_path)
            report["evaluation"] = eval_metrics

            if self.evaluator.should_rollback(eval_metrics):
                self._log("  性能下降，触发自动回滚")
                report["status"] = "rolled_back"
            else:
                self._log("  评估通过")
                report["status"] = "success"

            # 3.6: 缓存同步
            self._log("阶段3.6: 缓存同步清理")
            expired = self.cache.evict_expired()
            report["steps"].append(f"清理 {expired} 条过期缓存")

        except Exception as e:
            report["status"] = "error"
            report["errors"].append(str(e))
            self._log(f"  错误: {e}")

        report["duration_sec"] = time.time() - start
        report["timestamp"] = datetime.now().isoformat()

        self.run_history.append(report)
        self._save_report(report)

        return report

    def _log(self, msg: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [离线学习] {msg}"
        print(log_msg)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def _save_report(self, report: dict):
        """保存周期报告"""
        report_path = os.path.join(
            config.data_dir, "reports",
            f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def get_latest_report(self) -> Optional[dict]:
        """获取最近一次学习周期报告"""
        return self.run_history[-1] if self.run_history else None

    async def start_scheduled_loop(self):
        """启动定时循环（按配置间隔执行）"""
        import asyncio
        interval = config.offline_learning_interval_hours * 3600
        self._log(f"启动定时学习循环，间隔: {config.offline_learning_interval_hours} 小时")
        while True:
            await asyncio.sleep(interval)
            self._log("触发定时学习周期")
            await self.run_cycle()
