# 文档总览（Docs Index）

为了减少信息重复，项目文档按“入口 -> 专题”组织。

## 快速入口

- 新手启动：`README.md`
- 常用命令示例：`docs/USAGE_EXAMPLES.md`
- 配置说明：`docs/CONFIG_PROFILES.md`
- 发布流程：`RELEASE_PROCESS.md`

## 架构与工程

- 系统架构：`docs/ARCHITECTURE.md`
- 稳定性变更记录：`data/evolution/STABILITY_CHANGELOG.md`

## 运维与观测

- 运维排障：`docs/OPERATIONS_TROUBLESHOOTING.md`
- 观测看板 / 质量闸门 / 自动日报：`docs/OBSERVABILITY_PLAYBOOK.md`

## 建议阅读路径

1. `README.md`（先跑起来）
2. `docs/USAGE_EXAMPLES.md`（掌握日常命令）
3. `docs/ARCHITECTURE.md`（理解系统结构）
4. `docs/OBSERVABILITY_PLAYBOOK.md`（建立运维闭环）
5. `RELEASE_PROCESS.md`（发布与回滚）

## 维护原则

- 新增说明优先补充到专题文档，避免把 README 变成长文档。
- 同一主题只保留一个“权威说明”，其他文档使用链接引用。
- 自动生成产物（日报等）尽量保持最小样例，避免仓库膨胀。
