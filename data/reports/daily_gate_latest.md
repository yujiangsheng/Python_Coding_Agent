# PyCoder 每日报告

- 生成时间: 2026-02-16 15:00:37
- 项目路径: /Users/jiangshengyu/Documents/program/python/Python_Coding_Agent
- 统计窗口: 最近 10 轮

## 概览

- 健康检查: **PASS** (OK=7, FAIL=0, WARN=0)
- 当前轮次: 391
- 总体通过率: 0.7349
- 累计通过/总数: 815/1109
- Best score: 8.75 (round 220)

## 窗口指标

- 当前窗口: [382, 391] | rounds=10 | avg=0.9875 | std=0.0375 | full_pass=9
- 上一窗口: [372, 381] | rounds=10 | avg=0.9750 | std=0.0500 | full_pass=8
- 双 hard 任务 >=0.80: True

### 关键任务
- hard_calc_parser: pass_rate=1.0000, fails=0, flips=0, streak=2, delta_pass_rate=0.0000
- hard_concurrent_pool: pass_rate=0.8000, fails=1, flips=1, streak=4, delta_pass_rate=0.0222
- med_decorator_retry: pass_rate=1.0000, fails=0, flips=0, streak=2, delta_pass_rate=0.0000

## 质量闸门

- Gate 结果: **PASS**
- 阈值: avg>=0.9500, std<=0.0600, hard_task>=0.8000
- 当前: avg=0.9875, std=0.0375, parser=1.0000, pool=0.8000

## 健康检查明细

```text
== PyCoder Health Check ==
[OK]   Python: Python 3.9.6
[OK]   Files: 关键文件完整
[OK]   Configs: 配置文件可解析且关键字段完整
[OK]   Imports: 核心模块可导入
[OK]   WritableDirs: 关键数据目录可写
[OK]   Ollama: Ollama 可访问（models=7）
[OK]   系统健康检查通过
```

## 后续建议

- 当前状态可继续按既定节奏运行，建议每日保留一份报告归档。
