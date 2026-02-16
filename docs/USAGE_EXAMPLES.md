# 使用示例（Usage Examples）

以下示例覆盖最常见的使用路径。

## 1) 交互式 REPL

```bash
python main.py
```

示例输入：

```text
写一个支持 TTL 的线程安全 LRUCache，并给出最小测试。
```

## 2) 单次请求模式

```bash
python main.py --query "解释这段代码的时间复杂度并给优化建议"
```

## 3) 自我改进模式

```bash
python main.py --self-improve --iterations 2
```

适合离线维护阶段批量运行。

## 4) 演化验证模式

```bash
python evolve.py --resume --rounds 10
```

建议每个优化批次至少跑 8~10 轮窗口，并记录：

- 平均轮通过率
- 标准差（波动）
- 关键任务 pass_rate / fails / flips

## 5) REPL 命令示例

### 查看状态

```text
/status
```

### 查看记忆系统状态

```text
/memory
```

### 查看进化趋势

```text
/reflect
```

### 复杂任务编排

```text
/orchestrate
```

随后输入任务，例如：

```text
实现一个可扩展日志分析系统：包括解析、聚合、异常检测和可视化接口。
```

## 6) 发布流程示例

```bash
git add -A
git commit -m "release: v0.1.0"
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin main
git push origin v0.1.0
```

标签推送后会自动触发 `Release` workflow 创建 GitHub Release。

## 7) 扩展 REPL 命令（开发者示例）

`main.py` 已采用命令注册表模式。新增命令推荐两步：

1. 新增处理函数（示例）

```python
def _cmd_ping(agent: CodingAgent):
	print("\nPONG\n")
```

2. 在 `_command_registry()` 返回字典中注册

```python
"/ping": _cmd_ping,
```

这样无需继续扩展 `if/elif` 链，命令维护和测试都会更简单。
