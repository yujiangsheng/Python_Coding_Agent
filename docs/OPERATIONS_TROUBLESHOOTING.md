# 运维排障手册（Operations Troubleshooting）

本文档面向日常维护与线上排障，覆盖最常见故障场景：模型连接、进程异常、超时、内存压力、发布回滚、日志定位。

## 1. 快速健康检查

推荐先运行一键检查脚本：

```bash
python scripts/health_check.py
```

该脚本会检查：

- Python 版本
- 关键文件是否存在
- 配置 YAML 是否可解析
- 核心模块是否可导入
- 关键数据目录是否可写
- 若使用 Ollama，基础连通性是否正常

## 2. 常见故障与处理

### 2.1 模型不可用 / 无法生成

现象：
- 启动后长时间无响应
- 报错连接失败、后端不可用

排查：
1. 检查 `config.yaml` 中 `model.backend` 与 URL。
2. 若是 Ollama：确认服务是否运行。
3. 检查模型名是否存在（如 `qwen3-coder:30b`）。

建议：
- 本地先使用 `configs/config.dev.yaml` 降低开销验证。
- 生产再切换到 `configs/config.prod.yaml`。

### 2.2 演化流程卡住 / 超时

现象：
- `evolve.py --resume` 长时间无新日志
- 某轮 benchmark 多次 timeout

排查：
1. 查看 `data/evolution/evolution.log` 最近 200 行。
2. 检查是否存在僵尸进程或并发占用。
3. 确认 `execution.timeout` 与任务复杂度匹配。

可用操作（macOS/Linux）：

```bash
ps aux | grep "[e]volve.py"
```

必要时终止：

```bash
ps aux | grep "[e]volve.py" | awk '{print $2}' | xargs kill -9
```

### 2.3 进程被系统杀死（Exit 137/143）

含义：
- `137` 常见于 OOM 或 SIGKILL
- `143` 常见于 SIGTERM（被外部终止）

建议：
- 降低并发任务量或改用开发配置。
- 适当降低 `max_new_tokens`。
- 减少同时运行的模型/任务进程。

### 2.4 测试失败

基线检查：

```bash
python -m py_compile main.py evolve.py agent/core.py agent/code_generator.py
python test_agent.py
python test_memory_agent.py
python test_reflection_agent.py
```

处理建议：
- 先修编译错误，再修测试语义错误。
- 优先回滚最近一批改动，避免扩大故障面。

### 2.5 发布后回滚

推荐使用 `git revert` 回滚（避免强推）：

```bash
git revert <bad_commit_sha>
git push origin main
```

若是版本发布问题：
- 保留坏 tag 作为历史记录
- 新发修复 tag（如 `v0.1.1`）

## 3. 日志定位建议

重点日志：

- `data/agent.log`：主交互与模块错误
- `data/evolution/evolution.log`：演化轮次、任务通过率、失败摘要
- `data/evolution/evolution_state.json`：累计指标与状态快照

定位顺序：
1. 先看最近异常时间点。
2. 再看异常前后 50~100 行上下文。
3. 若为稳定性回归，对比最近两个窗口指标。

## 4. 稳定运行建议

- 开发阶段优先 `config.dev.yaml` 快速迭代。
- 合并前固定执行三套测试。
- 关键策略变更后至少跑 8~10 轮窗口验证。
- 将窗口证据更新到 `data/evolution/STABILITY_CHANGELOG.md`。

## 5. 升级与变更控制

推荐流程：
1. 小步提交（单一职责）。
2. 每次只改一个高风险组件。
3. 改后立即跑最小回归。
4. 达标后再扩展到长窗口验证。

这样可以显著降低排障成本与回滚复杂度。
