# 配置剖面说明（Config Profiles）

本项目提供 3 类配置入口：

- `config.yaml`：默认配置（通用）
- `configs/config.dev.yaml`：开发配置（迭代优先）
- `configs/config.prod.yaml`：生产/长稳配置（稳定优先）

## 一、如何切换配置

```bash
python main.py --config configs/config.dev.yaml
python main.py --config configs/config.prod.yaml
```

演化模式同理：

```bash
python evolve.py --resume --rounds 10
```

> 说明：`evolve.py` 当前读取默认配置路径；如需严格区分环境，建议在运行前切换默认配置文件，或后续扩展 `evolve.py` 的 `--config` 参数。

## 二、关键参数建议

### 1) model

- `backend`：建议本地优先 `ollama`
- `max_new_tokens`：
  - 开发：`4096~8192`（更快）
  - 生产：`8192~16384`（更稳）
- `temperature`：
  - 开发：`0.7`（探索）
  - 生产：`0.5~0.65`（稳定）

### 2) execution

- `timeout`：
  - 开发：可略低（20 秒）
  - 生产：建议 30 秒以上，降低误杀

### 3) self_improvement

- `min_confidence`：
  - 开发：`0.70~0.78`
  - 生产：`>=0.80`
- `max_iterations`：
  - 开发：`2~3`
  - 生产：`4~5`

### 4) reflection

- `quality_threshold`：
  - 开发：`0.60`
  - 生产：`0.65` 左右
- `llm_cooldown`：
  - 开发：可增加（降低调用成本）
  - 生产：保持较低（更及时反馈）

## 三、推荐工作流

1. 本地开发用 `config.dev.yaml` 快速迭代。
2. 每次合并前跑测试并进行一轮基准验证。
3. 发布前切到 `config.prod.yaml` 做稳定性复验。
4. 将关键窗口指标记录到 `data/evolution/STABILITY_CHANGELOG.md`。

## 四、排障提示

- 模型连接失败：检查 `ollama_url` 和本地模型是否已拉起。
- 反思调用频繁：提高 `reflection.llm_cooldown`。
- 自我改进风险偏高：提高 `self_improvement.min_confidence`。
- 记忆增长过快：降低 `memory.long_term.max_entries` 或提高去重阈值。
