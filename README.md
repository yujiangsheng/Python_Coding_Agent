# Python Coding Agent（PyCoder）

一个面向 Python 编程场景的中文智能体系统，支持：

- 代码生成 / 调试 / 解释 / 评审 / 测试生成
- 多层记忆（工作记忆、长期记忆、持久化记忆）
- 元知识提炼与自我改进
- 反思与进化追踪
- 基准演化循环（`evolve.py`）

## 快速开始

### 1) 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 配置模型

编辑 `config.yaml`（默认使用 Ollama）：

```yaml
model:
  backend: "ollama"
  ollama_model: "qwen3-coder:30b"
  ollama_url: "http://localhost:11434"
```

也可使用预设配置剖面：

```bash
python main.py --config configs/config.dev.yaml
python main.py --config configs/config.prod.yaml
```

### 3) 运行方式

交互模式：

```bash
python main.py
```

单次请求：

```bash
python main.py --query "写一个线程安全的LRU缓存"
```

自我改进模式：

```bash
python main.py --self-improve --iterations 3
```

演化基准模式：

```bash
python evolve.py --resume --rounds 10
```

## 常用 REPL 命令

- `/status`：查看系统状态
- `/memory`：查看记忆概览
- `/skills`：查看技能画像与短板
- `/meta`：触发元知识提炼
- `/orchestrate`：多智能体编排复杂任务
- `/reflect`：查看反思/进化趋势
- `/improve`：触发自我改进
- `/save`：保存会话和状态

## 系统架构（概览）

请求主链路：

1. 意图识别（`agent/intent.py`）
2. 记忆召回（`agent/memory/`）
3. 分发处理（`agent/core.py`）
4. 代码生成与执行（`agent/code_generator.py`）
5. 学习与反思（`agent/memory_agent.py`、`agent/reflection_agent.py`）

更详细结构见：`docs/ARCHITECTURE.md`

## 文档导航

- 架构说明：`docs/ARCHITECTURE.md`
- 使用示例：`docs/USAGE_EXAMPLES.md`
- 配置剖面：`docs/CONFIG_PROFILES.md`
- 运维排障：`docs/OPERATIONS_TROUBLESHOOTING.md`
- 观测看板：`docs/OBSERVABILITY_PLAYBOOK.md`
- 发布流程：`RELEASE_PROCESS.md`
- 演化稳定性记录：`data/evolution/STABILITY_CHANGELOG.md`

## 健康检查

```bash
python scripts/health_check.py
```

用于快速验证运行环境、配置完整性、核心模块导入与 Ollama 连通性。

窗口指标报告（JSON）：

```bash
python scripts/window_report.py --window 10 --compare-prev --key-tasks hard_calc_parser hard_concurrent_pool med_decorator_retry
```

## 开发与验证

```bash
python -m py_compile evolve.py main.py agent/code_generator.py agent/core.py
python test_agent.py
python test_memory_agent.py
python test_reflection_agent.py
```

## 适用场景

- 代码助手原型研发
- 本地大模型编程代理实践
- 记忆增强与长期学习机制实验
- 多智能体任务编排研究
