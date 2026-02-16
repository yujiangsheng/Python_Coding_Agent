# 系统架构说明（Architecture）

本文档面向维护者，解释模块职责、调用关系与关键数据流。

## 1. 顶层入口

- `main.py`
  - CLI 入口，支持 REPL / 单次查询 / 自我改进
  - 负责命令行参数解析与会话生命周期管理
- `evolve.py`
  - 演化循环入口：基准评测 → 反思 → 改进 → 测试 → 持久化

## 2. 核心编排层

- `agent/core.py`
  - 系统主控制器（`CodingAgent`）
  - 职责：
    1. 配置与组件懒加载
    2. 意图识别与处理分发
    3. RAG 记忆召回与上下文拼装
    4. 回答后学习与反思

## 3. 能力子系统

### 3.1 代码能力

- `agent/code_generator.py`
  - 代码生成、执行、自动修复、测试生成、代码评审

### 3.2 记忆系统

- `agent/memory/working_memory.py`：短期对话上下文
- `agent/memory/long_term_memory.py`：长期向量记忆
- `agent/memory/persistent_memory.py`：结构化持久记忆
- `agent/memory/external_memory.py`：外部检索
- `agent/memory/manager.py`：统一入口

### 3.3 学习/治理能力

- `agent/memory_agent.py`
  - 信息路由、错误注册、经验去重与维护建议
- `agent/reflection_agent.py`
  - 逐轮质量评估、会话回顾、进化趋势统计
- `agent/self_improver.py`
  - 提炼改进建议并尝试应用代码改动
- `agent/meta_knowledge.py`
  - 从经验中抽取高阶模式与元知识
- `agent/agent_orchestrator.py`
  - 拆解复杂任务，组织多子智能体协作

## 4. 核心数据流

## 4.1 聊天请求链路

1. 用户输入进入 `CodingAgent.chat()`
2. 写入工作记忆
3. 执行意图识别
4. 执行多层记忆召回（含外部搜索回退）
5. 根据意图路由到对应处理器
6. 写入回复并触发学习、反思
7. 若回复不确定，触发自动检索增强

## 4.2 演化链路（evolve）

1. 选择本轮活跃 benchmark
2. 调用智能体生成解答
3. 并行执行验证
4. 失败任务执行 validation-feedback retry
5. 汇总分数与通过率
6. 触发反思和自我改进
7. 执行测试回归并持久化状态

## 5. 关键设计原则

- 懒加载：降低启动成本
- 分层记忆：兼顾短期上下文与长期可检索知识
- 失败优先修复：把验证错误直接反馈到重试提示
- 可回滚演化：状态持久化 + 测试门禁
- 组件解耦：意图、代码、记忆、反思、改进分离

## 6. 维护建议

- 先保证 `test_*.py` 通过，再扩大改动范围
- 优先做“行为不变”的结构重构（例如提取共用函数、缓存映射）
- 变更演化策略时，始终记录窗口指标对比（通过率、波动、fails/flips）
