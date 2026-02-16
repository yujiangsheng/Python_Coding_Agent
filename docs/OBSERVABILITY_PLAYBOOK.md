# 观测与指标看板手册（Observability Playbook）

目标：让维护者能快速判断系统是否“健康、稳定、可发布”。

## 1. 关键 SLO（建议）

### 1.1 交互链路（agent）

- 可用性：关键命令可执行（`/status`、`/memory`、`/reflect`）
- 基线质量：三套测试持续通过
- 健康检查：`scripts/health_check.py` 全绿

### 1.2 演化链路（evolve）

- 窗口平均轮通过率（avg_round_pass_ratio） >= 0.95
- 窗口波动（std_round_pass_ratio） <= 0.06
- 关键 hard 任务通过率 >= 0.80（同一窗口）
- 满分轮占比持续稳定（full_pass_rounds / rounds）

> 上述阈值可按业务阶段调整，但建议保留“同窗口双 hard 任务达标”作为发布门槛。

## 2. 每日巡检清单

1. 运行健康检查：

```bash
python scripts/health_check.py
```

2. 采集最近窗口（示例：近 10 轮）：

```bash
python scripts/window_report.py --window 10 --key-tasks hard_calc_parser hard_concurrent_pool med_decorator_retry
```

3. 对比上一窗口（示例：前 10 轮 vs 当前 10 轮）：

```bash
python scripts/window_report.py --window 10 --compare-prev --key-tasks hard_calc_parser hard_concurrent_pool med_decorator_retry
```

4. 若指标回落：
- 优先查看 `data/evolution/evolution.log` 最近失败任务错误
- 检查对应任务的 retry 提示是否失效
- 小步修复后重跑 8~10 轮确认

5. 归档日报（Markdown）：

```bash
python scripts/daily_report.py --window 10
```

## 3. 发布前门禁

建议全部满足：

- 健康检查通过
- 三套测试通过
- 最新确认窗口满足：
  - 双 hard 任务通过率 >= 0.80
  - avg_round_pass_ratio >= 0.95
  - std_round_pass_ratio <= 0.06

## 4. 回归分析模板

建议每次策略变更后按模板记录：

- 变更摘要：
- 观察窗口：
- Overall：
  - avg_round_pass_ratio
  - std_round_pass_ratio
  - full_pass_rounds
- Key Tasks：
  - pass_rate / fails / flips / longest_pass_streak
- 结论：
  - 是否达成双 hard 任务门槛
  - 是否进入下一轮优化/发布

## 5. 关联文档

- 稳定性记录：`data/evolution/STABILITY_CHANGELOG.md`
- 运维排障：`docs/OPERATIONS_TROUBLESHOOTING.md`
- 配置剖面：`docs/CONFIG_PROFILES.md`
