# pairwise-punchin-monitor-latency

## 领域

游戏、图形与多媒体

## 主题

录音监听延迟补偿

## 说明

往返延迟测量、采样对齐与环缓栅栏。

初始状态下 `media/punchin_monitor_latency.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
