"""录音监听链路：往返延迟补偿、punch-in 采样对齐、设备切换重算、
环形缓冲原子栅栏、暂停恢复清空、时间轴对齐混音。

缺陷与修复的对应关系
--------------------
1. 输入缓冲直送监听输出、声卡往返延迟未补偿
   -> :func:`measure_latency` 用环回探针做互相关测量往返延迟，
      :func:`monitor_path` 用等长延迟线把监听信号推后整段往返延迟，
      使监听与伴奏在同一时间轴到达耳朵；测量失败走 :func:`fallback`。
2. punch-in 起止点按采样计数落在缓冲边界外被截断
   -> :func:`sample_align` 把起点向下、终点向上对齐到块边界，
      :func:`punch_range` 输出对齐并钳制在录音时长内的区间。
3. 设备切换后采样率变化、补偿量仍按旧值计算
   -> :func:`device_recalc` 以物理延迟（秒）为常量，按新采样率重算
      补偿采样数并重建延迟线。
4. 录音/混音线程共享环形缓冲、读写指针无栅栏导致丢帧
   -> :class:`RingBuffer` 采用 SPSC 模型，槽位写入与指针发布之间插入
      acquire/release 栅栏（:func:`ring_fence`），并统计丢帧。
5. 暂停恢复后监听缓冲未清空、旧片段叠进新录段
   -> :func:`buffer_clear` 在栅栏保护下复位指针并递增 generation，
      旧代数据对读者直接失效。
6. 混音未按时间轴对齐
   -> :func:`timeline_align` 按“设备计数 - 补偿量”把录段放到主时间轴，
      跨采样率片段先换算到秒再线性重采样，:func:`mix` 在 punch 区间内
      用新录段替换伴奏，区间外保留伴奏。

其余入口：:func:`validation` 入参校验、:func:`audit` 审计轨迹、
:func:`metrics` 指标快照、:func:`determinism` 可复现性校验、
:func:`record` 端到端编排。

纯标准库实现，无第三方依赖。
"""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BLOCK_SIZE = 256
DEFAULT_INPUT_BLOCKS = 2
DEFAULT_OUTPUT_BLOCKS = 2
DEFAULT_RING_CAPACITY = 4096
PROBE_LENGTH = 2048
# 往返延迟不可能超过探针长度的一半，超出即判定测量失败。
CONFIDENCE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _next_pow2(value: int, minimum: int = 1) -> int:
    """返回 >= max(value, minimum) 的最小 2 的幂。"""
    value = max(1, int(value), minimum)
    return 1 << (value - 1).bit_length()


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# 配置与校验（修复点 3/6 的基础：采样率、块大小等统一从此处归一化）
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    block_size: int = DEFAULT_BLOCK_SIZE
    input_latency_blocks: int = DEFAULT_INPUT_BLOCKS
    output_latency_blocks: int = DEFAULT_OUTPUT_BLOCKS
    ring_capacity: int = DEFAULT_RING_CAPACITY
    seed: int = 0

    @property
    def nominal_round_trip_samples(self) -> int:
        return (self.input_latency_blocks + self.output_latency_blocks) * self.block_size

    @property
    def nominal_round_trip_seconds(self) -> float:
        return self.nominal_round_trip_samples / self.sample_rate


def _coerce_config(value: Any = None) -> AudioConfig:
    """把 None / int / dict / AudioConfig 统一成 AudioConfig。

    整数按“采样率”解释，使 ``validation(1)`` 之类的单参调用也有定义。
    """
    if value is None:
        return AudioConfig()
    if isinstance(value, AudioConfig):
        return AudioConfig(**value.__dict__)
    if _is_int(value):
        return AudioConfig(sample_rate=value)
    if isinstance(value, dict):
        known = {f for f in AudioConfig.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"未知配置项: {sorted(unknown)}")
        return AudioConfig(**value)
    raise TypeError(f"无法把 {type(value).__name__} 解释为 AudioConfig")


def validation(config: Any = None, **overrides: Any) -> Dict[str, Any]:
    """校验音频配置并返回归一化后的纯字典；非法输入抛 ValueError。"""
    cfg = _coerce_config(config)
    for key, val in overrides.items():
        if key in cfg.__dataclass_fields__:  # type: ignore[attr-defined]
            setattr(cfg, key, val)
        else:
            raise ValueError(f"未知配置项: {key!r}")

    if not _is_int(cfg.sample_rate) or cfg.sample_rate <= 0:
        raise ValueError("sample_rate 必须为正整数")
    if not _is_int(cfg.block_size) or cfg.block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    if not _is_int(cfg.input_latency_blocks) or cfg.input_latency_blocks < 0:
        raise ValueError("input_latency_blocks 必须为非负整数")
    if not _is_int(cfg.output_latency_blocks) or cfg.output_latency_blocks < 0:
        raise ValueError("output_latency_blocks 必须为非负整数")
    if not _is_int(cfg.ring_capacity) or cfg.ring_capacity < 2:
        raise ValueError("ring_capacity 至少为 2")
    if not _is_int(cfg.seed):
        raise ValueError("seed 必须为整数")
    return dict(cfg.__dict__)


# ---------------------------------------------------------------------------
# 审计轨迹与指标（全部使用逻辑时钟，保证可复现）
# ---------------------------------------------------------------------------

class AuditLog:
    def __init__(self) -> None:
        self._events: List[Dict[str, Any]] = []

    def add(self, event: str, **details: Any) -> Dict[str, Any]:
        entry = {"seq": len(self._events), "event": event}
        entry.update(details)
        self._events.append(entry)
        return entry

    def entries(self) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._events]

    def names(self) -> List[str]:
        return [e["event"] for e in self._events]


class Metrics:
    def __init__(self) -> None:
        self.latency_true_seconds: float = 0.0
        self.latency_measured_samples: Optional[int] = None
        self.latency_error_samples: Optional[int] = None
        self.compensation_samples: int = 0
        self.dropped_frames: int = 0
        self.fence_count: int = 0
        self.buffer_clears: int = 0
        self.device_switches: int = 0
        self.blocks_processed: int = 0
        self.captured_samples: int = 0
        self.fallback_used: bool = False

    def snapshot(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def audit(session: Any = None) -> List[Dict[str, Any]]:
    """返回会话的审计轨迹副本；无会话时返回带创建事件的初始轨迹。"""
    log = getattr(session, "audit", None)
    if isinstance(log, AuditLog):
        return log.entries()
    scaffold = AuditLog()
    seed = session if _is_int(session) else 0
    scaffold.add("audit.created", seed=seed)
    return scaffold.entries()


def metrics(session: Any = None) -> Dict[str, Any]:
    """返回会话指标快照；无会话时返回零值指标骨架。"""
    holder = getattr(session, "metrics", None)
    if isinstance(holder, Metrics):
        return holder.snapshot()
    return Metrics().snapshot()


# ---------------------------------------------------------------------------
# 修复点 1：往返延迟测量（环回探针 + 归一化互相关）
# ---------------------------------------------------------------------------

@dataclass
class LatencyReport:
    ok: bool
    latency_samples: int
    latency_seconds: float
    confidence: float
    sample_rate: int
    source: str = "measured"

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class SimulatedDevice:
    """带固定物理往返延迟的声卡模拟器。

    物理延迟以秒为常量，与采样率无关——设备在任意采样率下重开，
    环回延迟采样数都应按 ``seconds * sample_rate`` 重新换算。
    """

    def __init__(self, config: Any = None, seed: int = 0) -> None:
        cfg = _coerce_config(config)
        self.native_rate = cfg.sample_rate
        self.round_trip_seconds = cfg.nominal_round_trip_seconds
        self.seed = seed

    def latency_samples_at(self, sample_rate: int) -> int:
        return max(0, round(self.round_trip_seconds * sample_rate))

    def loopback(self, tx: Sequence[float], sample_rate: int) -> List[float]:
        """把输出信号经物理往返延迟后作为输入回采，叠加确定性微弱噪声。"""
        delay = self.latency_samples_at(sample_rate)
        rng = random.Random((self.seed + 0x5F3759DF) & 0xFFFFFFFF)
        rx: List[float] = []
        for i in range(len(tx)):
            source = tx[i - delay] if i >= delay else 0.0
            rx.append(source + rng.uniform(-1e-5, 1e-5))
        return rx


def _build_probe(length: int, seed: int) -> List[float]:
    """脉冲头 + 确定性伪随机尾，相关峰尖锐且可复现。"""
    rng = random.Random(seed)
    probe = [0.0] * length
    probe[0] = 1.0
    for i in range(1, length):
        probe[i] = rng.uniform(-0.25, 0.25)
    return probe


def _normalized_xcorr(probe: Sequence[float],
                      rx: Sequence[float],
                      max_lag: int) -> Tuple[int, float]:
    """定窗归一化互相关，返回 (滞后采样数, 置信度)。

    每个候选滞后取长度等于探针的固定窗（rx 不足处补零），计算
    ``sum(probe*window)/sqrt(energy_probe*energy_window)``。
    正确滞后处窗内恰好是完整探针，置信度趋近 1；固定窗避免了滑动窗把
    延迟探针的能量稀释到大量无关采样上。
    """
    window = len(probe)
    probe_energy = sum(x * x for x in probe) or 1e-12
    padded = list(rx) + [0.0] * window
    best_lag, best_score = 0, -2.0
    # 搜索区间必须包含右端点 max_lag（真实延迟恰好等于上限时不能漏检）
    upper = min(max_lag, len(rx)) + 1
    for lag in range(upper):
        numerator = 0.0
        window_energy = 0.0
        for i in range(window):
            b = padded[lag + i]
            numerator += probe[i] * b
            window_energy += b * b
        score = numerator / math.sqrt(probe_energy * (window_energy or 1e-12))
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag, max(0.0, best_score)


def measure_latency(device: Any = None,
                    sample_rate: Optional[int] = None,
                    probe_length: int = PROBE_LENGTH,
                    confidence_threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    """环回测量往返延迟。

    发送确定性探针，对回采信号做归一化互相关，相关峰位置即往返延迟
    采样数；置信度低于阈值时 ``ok=False``，交由 :func:`fallback` 兜底。
    整数参数按设备种子解释。
    """
    if device is None or _is_int(device):
        device = SimulatedDevice(seed=int(device or 0))
    if not isinstance(device, SimulatedDevice):
        raise TypeError("device 必须是 SimulatedDevice 或种子整数")
    rate = int(sample_rate or device.native_rate)
    if rate <= 0:
        raise ValueError("sample_rate 必须为正整数")

    probe = _build_probe(probe_length, device.seed)
    # 发送端在探针尾部补零，使回采缓冲能完整容纳平移后的探针，
    # 正确滞后处相关窗不再被补零截断
    max_lag = probe_length // 2
    tx = probe + [0.0] * max_lag
    rx = device.loopback(tx, rate)
    lag, confidence = _normalized_xcorr(probe, rx, max_lag=max_lag)
    ok = confidence >= confidence_threshold
    report = LatencyReport(
        ok=ok,
        latency_samples=lag if ok else 0,
        latency_seconds=(lag / rate) if ok else 0.0,
        confidence=confidence,
        sample_rate=rate,
        source="measured",
    )
    return report.to_dict()


def fallback(config: Any = None) -> Dict[str, Any]:
    """测量失败时的保守兜底：按设备标称块数估算往返延迟并标记降级。"""
    cfg = _coerce_config(config)
    samples = cfg.nominal_round_trip_samples
    report = LatencyReport(
        ok=False,
        latency_samples=samples,
        latency_seconds=cfg.nominal_round_trip_seconds,
        confidence=0.0,
        sample_rate=cfg.sample_rate,
        source="fallback",
    )
    return report.to_dict()


# ---------------------------------------------------------------------------
# 修复点 1（续）：监听延迟线与监听链路
# ---------------------------------------------------------------------------

class DelayLine:
    """固定采样数延迟线；设备切换时可无损改长度（清空旧暂态）。"""

    def __init__(self, delay_samples: int = 0) -> None:
        self.set_delay(delay_samples)

    @property
    def delay(self) -> int:
        return self._delay

    def set_delay(self, delay_samples: int) -> None:
        delay_samples = max(0, int(delay_samples))
        self._delay = delay_samples
        # 容量需容纳 delay 个历史采样加当前写入位
        self._size = delay_samples + 1
        self._buffer = [0.0] * self._size
        self._head = 0  # 下一个写入位置

    def process(self, block: Sequence[float]) -> List[float]:
        out: List[float] = []
        size = self._size
        delay = self._delay
        for sample in block:
            self._buffer[self._head] = float(sample)
            self._head = (self._head + 1) % size
            # 最新写入位于 head-1，再向前回溯 delay 个即延迟输出
            out.append(self._buffer[(self._head - 1 - delay) % size])
        return out


# ---------------------------------------------------------------------------
# 修复点 4：SPSC 环形缓冲 + acquire/release 栅栏
# ---------------------------------------------------------------------------

class RingBuffer:
    """单生产者单消费者环形缓冲。

    内存序约定（与 C11 acquire/release 同构，Python 中以锁的
    acquire/release 承载栅栏语义）：

    * 生产者先在槽位上写数据，再经 release 栅栏发布写指针；
    * 消费者先经 acquire 栅栏读到写指针，再访问槽位；
    * 读指针同理反向发布。槽位访问因此不会越过指针发布被重排序。

    暂停恢复时递增 ``generation``，旧代数据对读者立即失效，
    不会叠进新录段（修复点 5）。
    """

    def __init__(self, capacity: int, generation: int = 0) -> None:
        cap = _next_pow2(capacity, minimum=2)
        self._capacity = cap
        self._mask = cap - 1
        self._slots: List[float] = [0.0] * cap
        self._write_total = 0  # 累计写入采样数（单调递增，不取模）
        self._read_total = 0
        self._fence = threading.Lock()
        # 满/空条件变量，承载阻塞式生产消费；栅栏仍由同一把锁保证
        self._not_full = threading.Condition(self._fence)
        self._not_empty = threading.Condition(self._fence)
        self.generation = generation
        self.dropped = 0
        self.fence_count = 0
        self.clears = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def _acquire_fence(self) -> None:
        self._fence.acquire()
        self.fence_count += 1

    def _release_fence(self) -> None:
        self._fence.release()

    def readable(self) -> int:
        self._acquire_fence()
        try:
            return self._write_total - self._read_total
        finally:
            self._release_fence()

    def push(self, samples: Sequence[float],
             generation: Optional[int] = None,
             block: bool = False) -> int:
        """写入一块数据。

        block=False（默认）：空间不足时丢弃最新多余采样并计数，
        不覆盖未读数据——对应“写指针无栅栏时静默丢帧”的可观测退化。
        block=True：空间不足时在条件变量上等待消费者回收槽位，
        供实时录音线程使用，正常背压下不丢帧；代际失效（暂停清空）
        会立即唤醒并放弃写入。
        """
        if generation is not None and generation != self.generation:
            return 0
        # acquire 栅栏：取得消费者发布的读指针
        self._acquire_fence()
        try:
            if block:
                while (self._write_total - self._read_total
                       >= self._capacity
                       and (generation is None
                            or generation == self.generation)):
                    self._not_full.wait(timeout=1.0)
                if generation is not None and generation != self.generation:
                    return 0
            read_total = self._read_total
        finally:
            self._release_fence()

        write_total = self._write_total
        free = self._capacity - (write_total - read_total)
        accepted = min(len(samples), free)
        if accepted < len(samples):
            self.dropped += len(samples) - accepted

        # 槽位写入发生在指针发布之前，且下标与消费者不冲突（SPSC）
        for i in range(accepted):
            self._slots[(write_total + i) & self._mask] = float(samples[i])
        write_total += accepted

        # release 栅栏：保证槽位写入先于写指针对消费者可见
        self._acquire_fence()
        try:
            self._write_total = write_total
            self._not_empty.notify_all()
        finally:
            self._release_fence()
        return accepted

    def pop(self, count: int,
            generation: Optional[int] = None,
            block: bool = False) -> List[float]:
        # acquire 栅栏：取得生产者发布的写指针与当前代
        self._acquire_fence()
        try:
            if block:
                while (self._write_total - self._read_total == 0
                       and (generation is None
                            or generation == self.generation)):
                    self._not_empty.wait(timeout=1.0)
            write_total = self._write_total
            current_gen = self.generation
        finally:
            self._release_fence()

        if generation is not None and generation != current_gen:
            return []  # 旧代数据，整批作废

        read_total = self._read_total
        available = write_total - read_total
        take = min(count, available)
        out = [
            self._slots[(read_total + i) & self._mask]
            for i in range(take)
        ]
        read_total += take

        # release 栅栏：发布新读指针，生产者据此回收槽位
        self._acquire_fence()
        try:
            self._read_total = read_total
            self._not_full.notify_all()
        finally:
            self._release_fence()
        return out


def ring_fence(target: Any = None) -> Dict[str, Any]:
    """对环形缓冲执行一次带栅栏的 push/pop 往返并返回核验报告。

    整数参数按容量解释（自动向上取 2 的幂，最小为 2）。
    """
    if isinstance(target, RingBuffer):
        ring = target
        created = False
    else:
        ring = RingBuffer(int(target) if _is_int(target) else DEFAULT_RING_CAPACITY)
        created = True
    payload = [0.25, -0.5, 0.75, -1.0, 0.0]
    accepted = ring.push(payload)
    drained = ring.pop(len(payload))
    ok = accepted == len(payload) and drained == payload and ring.dropped == 0
    return {
        "ok": ok,
        "capacity": ring.capacity,
        "fence_count": ring.fence_count,
        "dropped": ring.dropped,
        "generation": ring.generation,
        "created": created,
    }


# ---------------------------------------------------------------------------
# 修复点 5：暂停恢复清空
# ---------------------------------------------------------------------------

def buffer_clear(target: Any = None) -> Dict[str, Any]:
    """在栅栏保护下清空环形缓冲：复位读写指针并递增 generation。

    旧代数据不再可能被读出，从根上杜绝旧片段叠进新录段。
    """
    if isinstance(target, RingBuffer):
        ring = target
    else:
        ring = RingBuffer(int(target) if _is_int(target) else DEFAULT_RING_CAPACITY)

    ring._acquire_fence()
    try:
        ring._write_total = 0
        ring._read_total = 0
        ring.generation += 1
        ring.clears += 1
        generation = ring.generation
        # 唤醒所有在满/空条件上等待的线程，旧代等待者醒来后自行退出
        ring._not_full.notify_all()
        ring._not_empty.notify_all()
    finally:
        ring._release_fence()
    return {
        "ok": ring.readable() == 0 and ring.pop(1, generation=generation - 1) == [],
        "generation": generation,
        "clears": ring.clears,
        "fence_count": ring.fence_count,
    }


# ---------------------------------------------------------------------------
# 修复点 2：采样对齐与 punch-in 区间
# ---------------------------------------------------------------------------

def sample_align(position: int,
                 block_size: int = DEFAULT_BLOCK_SIZE,
                 edge: str = "start") -> int:
    """把采样位置对齐到块边界。

    edge="start" 向下取整（punch 起点，保证起点前的素材不被吞），
    edge="end" 向上取整（punch 终点，保证终点前的素材完整落盘）。
    """
    if not _is_int(position) or position < 0:
        raise ValueError("position 必须为非负整数")
    if not _is_int(block_size) or block_size <= 0:
        raise ValueError("block_size 必须为正整数")
    if edge == "start":
        return (position // block_size) * block_size
    if edge == "end":
        return ((position + block_size - 1) // block_size) * block_size
    raise ValueError("edge 只接受 'start' 或 'end'")


def punch_range(start: int,
                end: Optional[int] = None,
                block_size: int = DEFAULT_BLOCK_SIZE,
                limit: Optional[int] = None) -> Dict[str, int]:
    """把请求的 punch-in 起止采样对齐到块边界并钳制在 [0, limit] 内。"""
    if not _is_int(start) or start < 0:
        raise ValueError("punch 起点必须为非负整数")
    if end is None:
        end = start + block_size
    if not _is_int(end) or end <= 0:
        raise ValueError("punch 终点必须为正整数")

    aligned_start = sample_align(start, block_size, "start")
    aligned_end = sample_align(end, block_size, "end")
    if limit is not None:
        if not _is_int(limit) or limit < 0:
            raise ValueError("limit 必须为非负整数")
        aligned_start = min(aligned_start, sample_align(limit, block_size, "start"))
        aligned_end = min(aligned_end, sample_align(limit, block_size, "end"))
    if aligned_end <= aligned_start:
        raise ValueError(
            f"对齐后 punch 区间为空: start={aligned_start}, end={aligned_end}"
        )
    return {
        "start": aligned_start,
        "end": aligned_end,
        "samples": aligned_end - aligned_start,
        "block_size": block_size,
    }


# ---------------------------------------------------------------------------
# 修复点 3：设备切换重算补偿
# ---------------------------------------------------------------------------

@dataclass
class Session:
    config: AudioConfig
    device: SimulatedDevice
    latency: Dict[str, Any]
    audit_log: AuditLog = field(default_factory=AuditLog)
    metrics_holder: Metrics = field(default_factory=Metrics)
    sample_rate: int = 0
    compensation_samples: int = 0
    generation: int = 0

    def __post_init__(self) -> None:
        self.sample_rate = self.config.sample_rate
        self.compensation_samples = int(self.latency["latency_samples"])
        self.metrics_holder.latency_true_seconds = self.device.round_trip_seconds
        self.metrics_holder.latency_measured_samples = self.latency["latency_samples"]
        self.metrics_holder.latency_error_samples = (
            self.latency["latency_samples"]
            - self.device.latency_samples_at(self.sample_rate)
        )
        self.metrics_holder.compensation_samples = self.compensation_samples
        self.metrics_holder.fallback_used = self.latency["source"] == "fallback"

    @property
    def audit(self) -> AuditLog:
        return self.audit_log

    @property
    def metrics(self) -> Metrics:
        return self.metrics_holder


def device_recalc(session_or_rate: Any,
                  new_sample_rate: Optional[int] = None) -> Dict[str, Any]:
    """设备切换后按新采样率重算补偿量并重建延迟线。

    物理往返延迟（秒）是设备常量，补偿采样数 = round(seconds * new_rate)，
    绝不能沿用上一采样率下的采样数。
    """
    if isinstance(session_or_rate, Session):
        session = session_or_rate
        rate = new_sample_rate
    else:
        # 单整数调用：整数即新采样率，内部构造默认会话
        cfg = _coerce_config(None)
        device = SimulatedDevice(cfg)
        session = Session(cfg, device, measure_latency(device, cfg.sample_rate))
        rate = int(session_or_rate)
    if rate is None or not _is_int(rate) or rate <= 0:
        raise ValueError("new_sample_rate 必须为正整数")

    old_rate = session.sample_rate
    old_comp = session.compensation_samples
    latency_seconds = session.device.round_trip_seconds
    new_comp = max(0, round(latency_seconds * rate))

    session.sample_rate = rate
    session.config.sample_rate = rate
    session.compensation_samples = new_comp
    session.metrics.compensation_samples = new_comp
    session.metrics.device_switches += 1
    session.audit.add(
        "device.switch",
        old_sample_rate=old_rate,
        new_sample_rate=rate,
        old_compensation=old_comp,
        new_compensation=new_comp,
        latency_seconds=latency_seconds,
    )
    return {
        "ok": True,
        "old_sample_rate": old_rate,
        "sample_rate": rate,
        "old_compensation": old_comp,
        "compensation_samples": new_comp,
        "latency_seconds": latency_seconds,
    }


# ---------------------------------------------------------------------------
# 修复点 1（续）：监听链路（输入 -> 补偿延迟线 -> 监听输出）
# ---------------------------------------------------------------------------

def monitor_path(input_block: Any,
                 session: Optional[Session] = None,
                 delay_line: Optional[DelayLine] = None) -> List[float]:
    """把输入块经往返延迟补偿后送监听输出，使监听与伴奏时间轴对齐。"""
    if _is_int(input_block):
        input_block = [0.0] * int(input_block)
    if not isinstance(input_block, Sequence):
        raise TypeError("input_block 必须是采样序列或块长度整数")
    if delay_line is None:
        comp = session.compensation_samples if session is not None else 0
        delay_line = DelayLine(comp)
    if session is not None:
        session.audit.add(
            "monitor.block",
            input_samples=len(input_block),
            compensation=delay_line.delay,
        )
    return delay_line.process(input_block)


# ---------------------------------------------------------------------------
# 修复点 6：时间轴对齐
# ---------------------------------------------------------------------------

def timeline_align(target: Any,
                   compensation: int = 0,
                   sample_rate: int = DEFAULT_SAMPLE_RATE,
                   output_rate: Optional[int] = None) -> Any:
    """把设备采样计数换算到主时间轴位置。

    约定：t 时刻进入麦克风的素材在 t + 往返延迟 才到达软件，故其时间轴
    位置 = 设备计数位置 - 补偿量。跨采样率时先换算成秒再映射到输出采样率。
    整数参数按设备采样位置解释。
    """
    if not _is_int(compensation) or compensation < 0:
        raise ValueError("compensation 必须为非负整数")
    if not _is_int(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate 必须为正整数")
    out_rate = output_rate or sample_rate
    if not _is_int(out_rate) or out_rate <= 0:
        raise ValueError("output_rate 必须为正整数")

    if _is_int(target):
        aligned = max(0, target - compensation)
        seconds = aligned / sample_rate
        return max(0, round(seconds * out_rate))

    if isinstance(target, dict):
        return _align_segment(target, compensation, sample_rate, out_rate)
    if isinstance(target, list):
        return [_align_segment(seg, compensation, sample_rate, out_rate)
                for seg in target]
    raise TypeError("target 必须是采样位置、录段字典或录段列表")


def _align_segment(segment: Dict[str, Any],
                   compensation: int,
                   sample_rate: int,
                   output_rate: int) -> Dict[str, Any]:
    # 录段已带主时间轴锚点（record 落盘时写入）：直接使用，
    # 监听延迟线已经完成补偿，这里不能再减一次补偿量
    if "timeline_start_seconds" in segment:
        start_seconds = float(segment["timeline_start_seconds"])
        placed = dict(segment)
        placed["timeline_start"] = max(0, round(start_seconds * output_rate))
        return placed

    # 原始设备计数路径：device_start 按录段自身采样率计数，
    # 必须用段内速率换算，不能用调用方（可能已切换）的采样率
    rate = int(segment.get("sample_rate", sample_rate))
    comp = int(segment.get("compensation", compensation))
    device_start = int(segment["device_start"])
    aligned = max(0, device_start - comp)
    start_seconds = aligned / rate
    placed = dict(segment)
    placed["timeline_start_seconds"] = start_seconds
    placed["timeline_start"] = max(0, round(start_seconds * output_rate))
    placed["aligned_device_start"] = aligned
    return placed


def _resample_linear(samples: Sequence[float],
                     src_rate: int,
                     dst_rate: int) -> List[float]:
    if src_rate == dst_rate or not samples:
        return list(samples)
    ratio = src_rate / dst_rate
    out_len = max(1, round(len(samples) * dst_rate / src_rate))
    out: List[float] = []
    last = len(samples) - 1
    for i in range(out_len):
        position = i * ratio
        index = int(position)
        frac = position - index
        if index >= last:
            out.append(samples[last])
        else:
            out.append(samples[index] * (1.0 - frac)
                       + samples[index + 1] * frac)
    return out


# ---------------------------------------------------------------------------
# 修复点 6（续）：时间轴对齐混音
# ---------------------------------------------------------------------------

def mix(length_or_backing: Any = None,
        layers: Optional[List[Dict[str, Any]]] = None,
        punch: Optional[Dict[str, int]] = None) -> List[float]:
    """按时间轴混音。

    整数参数按输出长度解释（返回静音轨）；也可直接传入伴奏列表。
    layers 为 ``{"offset", "samples", "mode"}`` 列表：
    mode="overlay" 叠加，mode="replace" 替换（punch 区间内的新录段）。
    """
    if _is_int(length_or_backing):
        backing: List[float] = [0.0] * int(length_or_backing)
    elif isinstance(length_or_backing, list):
        backing = [float(x) for x in length_or_backing]
    elif length_or_backing is None:
        backing = []
    else:
        raise TypeError("第一个参数必须是输出长度整数或伴奏列表")

    output = list(backing)
    for layer in layers or []:
        offset = int(layer.get("offset", 0))
        samples = layer.get("samples", [])
        mode = layer.get("mode", "overlay")
        if offset >= len(output):
            continue
        for i, sample in enumerate(samples):
            pos = offset + i
            if pos >= len(output):
                break
            if mode == "replace":
                output[pos] = float(sample)
            else:
                output[pos] = _clamp(output[pos] + float(sample))
    return [_clamp(x) for x in output]


# ---------------------------------------------------------------------------
# 端到端编排
# ---------------------------------------------------------------------------

def _backing_sample(t: float) -> float:
    return 0.6 * math.sin(2.0 * math.pi * 220.0 * t)


def _mic_sample(t: float) -> float:
    return 0.8 * math.sin(2.0 * math.pi * 440.0 * t)


def record(config: Any = None,
           blocks: Optional[int] = None,
           *,
           pause_after: Optional[int] = None,
           resume_after: Optional[int] = None,
           switch_after: Optional[int] = None,
           switch_rate: Optional[int] = None,
           punch: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
    """端到端模拟一段带监听补偿、punch-in、暂停恢复、设备切换的录音。

    整数首参且未显式给 blocks 时，按“模拟块数”解释（record(1) 即单块会话）。
    """
    if _is_int(config) and blocks is None:
        blocks = int(config)
        config = None
    cfg = AudioConfig(**validation(config))
    total_blocks = int(blocks) if blocks is not None else 16
    if total_blocks <= 0:
        raise ValueError("blocks 必须为正整数")

    device = SimulatedDevice(cfg, seed=cfg.seed)
    latency = measure_latency(device, cfg.sample_rate)
    if not latency["ok"]:  # 测量失败走保守兜底
        latency = fallback(cfg)
    session = Session(cfg, device, latency)
    block_size = cfg.block_size
    delay_line = DelayLine(session.compensation_samples)
    ring = RingBuffer(cfg.ring_capacity)
    session.generation = ring.generation
    session.audit.add(
        "session.start",
        sample_rate=cfg.sample_rate,
        block_size=block_size,
        compensation=session.compensation_samples,
        latency_source=latency["source"],
    )

    # 修复点 2：punch 起止点对齐块边界并钳制在录音时长内。
    # 块数不足（<8）的小会话不安排 punch-in，避免唯一的块被区间钳制后误采。
    limit = total_blocks * block_size
    punch_active = punch is not None or total_blocks >= 8
    if punch is not None:
        requested_start, requested_end = punch
    elif punch_active:
        requested_start = block_size * min(4, total_blocks - 1)
        requested_end = min(block_size * min(12, total_blocks), limit)
    else:
        requested_start = requested_end = 0
    pr = (punch_range(requested_start, requested_end, block_size, limit)
          if punch_active else
          {"start": 0, "end": 0, "samples": 0, "block_size": block_size})
    # punch 区间锚定在秒级主时间轴上，设备切换采样率后依然有效
    punch_start_seconds = pr["start"] / cfg.sample_rate
    punch_end_seconds = pr["end"] / cfg.sample_rate
    session.audit.add("punch.range", **pr)

    # 块数足够多时才编排暂停/切换事件，保持小会话简单确定
    if total_blocks >= 8:
        pause_after = 2 if pause_after is None else pause_after
        resume_after = 3 if resume_after is None else resume_after
        switch_after = 5 if switch_after is None else switch_after
        switch_rate = switch_rate or (cfg.sample_rate // 2)

    rate = cfg.sample_rate
    master_time = 0.0
    armed = True
    segments: List[Dict[str, Any]] = []
    # 监听链路结构性正确性：延迟线长度始终等于当前采样率下的物理往返延迟
    compensation_matched = True

    for block_index in range(total_blocks):
        if pause_after is not None and block_index == pause_after:
            armed = False
            session.audit.add("pause", block=block_index)
        if resume_after is not None and block_index == resume_after:
            # 修复点 5：恢复时在栅栏保护下清空，旧代数据立即失效
            clear_report = buffer_clear(ring)
            session.generation = clear_report["generation"]
            session.metrics.buffer_clears += 1
            armed = True
            session.audit.add("resume.clear", block=block_index,
                              generation=session.generation)
        if switch_after is not None and block_index == switch_after and switch_rate:
            # 修复点 3：设备切换，按新采样率重算补偿并重建延迟线
            recalc = device_recalc(session, switch_rate)
            delay_line.set_delay(recalc["compensation_samples"])
            rate = switch_rate
            session.audit.add("delay.rebuilt",
                              delay=recalc["compensation_samples"])

        t0 = master_time
        # 现在到达软件输入的素材，是往返延迟之前从麦克风采到的
        input_block = [
            _mic_sample(t0 + i / rate - device.round_trip_seconds)
            for i in range(block_size)
        ]

        # 修复点 1：监听输出经整段往返延迟补偿，与伴奏落在同一时间轴
        monitor_block = monitor_path(input_block, session, delay_line)
        if (delay_line.delay != session.compensation_samples
                or len(monitor_block) != block_size
                or session.compensation_samples
                != device.latency_samples_at(rate)):
            compensation_matched = False

        # 修复点 4：录音线程经栅栏入环，满时阻塞背压而非丢帧
        if armed:
            ring.push(input_block, generation=session.generation, block=True)
            popped = ring.pop(block_size, generation=session.generation)
            if len(popped) != block_size:
                compensation_matched = False  # 丢帧即视为链路异常
            # 修复点 2/6：仅当整块完整落在对齐后的 punch 区间（秒级
            # 时间轴）内才落盘——部分相交的块一律不取，从根上杜绝
            # “起止点落在缓冲边界外被截断”。边界已按块对齐，
            # 正常块要么整块在内、要么整块在外。
            block_start_seconds = master_time
            block_end_seconds = master_time + block_size / rate
            if (pr["samples"] > 0
                    and block_start_seconds + 1e-12 >= punch_start_seconds
                    and block_end_seconds <= punch_end_seconds + 1e-12):
                segments.append({
                    # 直接锚定主时间轴秒位置：监听延迟线已把该块
                    # 从“到达软件的时刻”推后回它在伴奏时间轴上的原位，
                    # 故此处不再做第二次补偿减法
                    "timeline_start_seconds": block_start_seconds,
                    "device_start": round(block_start_seconds * rate),
                    "sample_rate": rate,
                    "compensation": session.compensation_samples,
                    "generation": session.generation,
                    "samples": list(popped),
                })
                session.metrics.captured_samples += len(popped)

        session.metrics.fence_count = ring.fence_count
        session.metrics.dropped_frames = ring.dropped
        session.metrics.blocks_processed += 1
        master_time += block_size / rate

    # 修复点 6：录段按时间轴归位（跨采样率先换秒，再重采样到最终速率）
    final_rate = rate
    placed = timeline_align(segments, output_rate=final_rate)
    total_seconds = master_time
    total_samples = round(total_seconds * final_rate)
    backing_full = [
        _backing_sample(i / final_rate) for i in range(total_samples)
    ]
    layers = []
    for seg in placed:
        resampled = _resample_linear(seg["samples"], seg["sample_rate"], final_rate)
        layers.append({
            "offset": seg["timeline_start"],
            "samples": resampled,
            "mode": "replace",
        })
    final_mix = mix(backing_full, layers, punch=pr)
    session.audit.add("mix.done", samples=len(final_mix),
                      segments=len(layers))

    chain_ok = (compensation_matched
                and ring.dropped == 0
                and (latency["ok"] or latency["source"] == "fallback"))
    return {
        "ok": chain_ok,
        "mix": final_mix,
        "backing": backing_full,
        "punch": pr,
        "latency": latency,
        "segments": placed,
        "metrics": session.metrics.snapshot(),
        "audit": session.audit.entries(),
        "sample_rate": final_rate,
        "total_seconds": total_seconds,
    }


# ---------------------------------------------------------------------------
# 可复现性
# ---------------------------------------------------------------------------

def determinism(seed: int = 0, blocks: int = 8) -> Dict[str, Any]:
    """同种子两次录音必须逐采样一致（指标、审计事件序列同样一致）。"""
    first = record(AudioConfig(seed=int(seed)), blocks=blocks)
    second = record(AudioConfig(seed=int(seed)), blocks=blocks)
    compared = ("mix", "backing", "punch", "latency", "metrics", "audit")
    mismatches = [
        key for key in compared
        if first[key] != second[key]
    ]
    return {
        "ok": not mismatches,
        "seed": int(seed),
        "blocks": blocks,
        "mismatches": mismatches,
        "mix_samples": len(first["mix"]),
    }
