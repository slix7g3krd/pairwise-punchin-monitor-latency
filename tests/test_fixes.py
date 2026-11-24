"""六类缺陷修复的行为回归测试（不依赖官方契约测试的弱断言）。

覆盖：
1. 往返延迟测量 + 监听补偿（measure_latency / fallback / monitor_path）
2. punch-in 起止点采样对齐（sample_align / punch_range）
3. 设备切换按新采样率重算补偿（device_recalc）
4. 环形缓冲 acquire/release 栅栏、并发不丢帧（ring_fence / RingBuffer）
5. 暂停恢复清空、旧代数据失效（buffer_clear）
6. 时间轴对齐混音（timeline_align / mix / record）
另含可复现性、入参校验与端到端编排。
"""

import threading

import pytest

import media.punchin_monitor_latency as mod


# ---------------------------------------------------------------------------
# 修复 1：往返延迟测量与监听补偿
# ---------------------------------------------------------------------------

def test_measure_latency_finds_true_round_trip():
    device = mod.SimulatedDevice(mod.AudioConfig(seed=11), seed=11)
    report = mod.measure_latency(device, 48000)
    assert report["ok"] is True
    assert report["latency_samples"] == device.latency_samples_at(48000)
    assert report["latency_seconds"] == pytest.approx(
        device.round_trip_seconds, abs=1e-9)
    assert report["confidence"] > 0.99


def test_measure_latency_search_range_includes_upper_bound():
    # 真实延迟恰好等于搜索上限时也必须命中（回归 range 右端点漏检）
    config = mod.AudioConfig(seed=5, input_latency_blocks=2,
                             output_latency_blocks=2, block_size=256)
    device = mod.SimulatedDevice(config, seed=5)
    # 往返延迟 = 4 * 256 = 1024，恰为默认探针 2048 的一半（搜索上限）
    assert device.latency_samples_at(48000) == 1024
    report = mod.measure_latency(device, 48000)
    assert report["ok"] and report["latency_samples"] == 1024


def test_low_confidence_measurement_reports_failure():
    device = mod.SimulatedDevice(mod.AudioConfig(seed=1), seed=1)
    report = mod.measure_latency(
        device, 48000, probe_length=8, confidence_threshold=0.99)
    assert report["ok"] is False
    assert report["source"] == "measured"


def test_fallback_uses_nominal_latency_and_flags_degradation():
    report = mod.fallback(48000)
    cfg = mod.AudioConfig(sample_rate=48000)
    assert report["ok"] is False
    assert report["source"] == "fallback"
    assert report["latency_samples"] == cfg.nominal_round_trip_samples
    assert report["latency_seconds"] == pytest.approx(
        cfg.nominal_round_trip_seconds)


def test_delay_line_semantics_and_cross_block_history():
    assert mod.DelayLine(0).process([1.0, 2.0]) == [1.0, 2.0]
    assert mod.DelayLine(3).process(
        [1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.0, 0.0, 1.0, 2.0]
    line = mod.DelayLine(2)
    assert line.process([1.0, 2.0]) == [0.0, 0.0]
    assert line.process([3.0]) == [1.0]  # 流 1,2,3 延迟 2 -> 0,0,1


def test_monitor_path_applies_round_trip_compensation():
    out = mod.monitor_path([1.0, 2.0, 3.0], None, mod.DelayLine(1))
    assert out == [0.0, 1.0, 2.0]
    # 整数按块长度解释，返回等长零初始块
    assert len(mod.monitor_path(4)) == 4


# ---------------------------------------------------------------------------
# 修复 2：punch-in 采样对齐
# ---------------------------------------------------------------------------

def test_sample_align_start_floors_end_ceils_to_block_boundary():
    assert mod.sample_align(0, 256, "start") == 0
    assert mod.sample_align(300, 256, "start") == 256
    assert mod.sample_align(300, 256, "end") == 512
    assert mod.sample_align(256, 256, "end") == 256  # 已在边界上


def test_sample_align_rejects_bad_inputs():
    with pytest.raises(ValueError):
        mod.sample_align(-1)
    with pytest.raises(ValueError):
        mod.sample_align(10, block_size=0)
    with pytest.raises(ValueError):
        mod.sample_align(10, edge="middle")


def test_punch_range_aligns_and_clamps_to_limit():
    pr = mod.punch_range(300, 700, 256, 1000)
    assert pr == {"start": 256, "end": 768, "samples": 512,
                  "block_size": 256}
    # 钳制：请求超出录音时长
    clamped = mod.punch_range(0, 5000, 256, 1024)
    assert clamped["end"] <= 1024
    assert clamped["start"] == 0


def test_punch_range_rejects_empty_interval():
    # 同一块内的起点和终点：起点向下对齐到块首，终点向上对齐到下一块首，
    # 这里构造“起点已在块首、终点未越过该块”的退化情形 -> 对齐后首尾重合
    with pytest.raises(ValueError):
        mod.punch_range(512, 512, 256)
    with pytest.raises(ValueError):
        mod.punch_range(512, 256, 256)  # 终点不大于起点


# ---------------------------------------------------------------------------
# 修复 3：设备切换重算补偿
# ---------------------------------------------------------------------------

def test_device_recalc_recomputes_compensation_from_physical_seconds():
    device = mod.SimulatedDevice(mod.AudioConfig(seed=1), seed=1)
    session = mod.Session(
        mod.AudioConfig(seed=1), device,
        mod.measure_latency(device, 48000))
    assert session.compensation_samples == 1024
    report = mod.device_recalc(session, 24000)
    assert report["old_sample_rate"] == 48000
    assert report["sample_rate"] == 24000
    assert report["old_compensation"] == 1024
    assert report["compensation_samples"] == 512
    # 物理延迟（秒）恒定，采样数随采样率等比变化
    assert report["compensation_samples"] / 24000 == pytest.approx(
        report["latency_seconds"], abs=1e-12)
    assert session.sample_rate == 24000
    assert session.metrics.device_switches == 1
    assert any(e["event"] == "device.switch" for e in session.audit.entries())


def test_device_recalc_integer_form_stands_alone():
    report = mod.device_recalc(96000)
    assert report["ok"] and report["sample_rate"] == 96000
    assert report["compensation_samples"] > report["old_compensation"]


def test_device_recalc_rejects_non_positive_rate():
    device = mod.SimulatedDevice(mod.AudioConfig(seed=1), seed=1)
    session = mod.Session(
        mod.AudioConfig(seed=1), device,
        mod.measure_latency(device, 48000))
    with pytest.raises(ValueError):
        mod.device_recalc(session, 0)


# ---------------------------------------------------------------------------
# 修复 4：环形缓冲栅栏
# ---------------------------------------------------------------------------

def test_ring_fence_roundtrip_preserves_samples():
    ring = mod.RingBuffer(16)
    report = mod.ring_fence(ring)
    assert report["ok"] is True
    assert report["dropped"] == 0
    assert report["fence_count"] >= 4  # push/pop 各自的 acquire+release


def test_ring_buffer_drops_on_overflow_without_overwriting():
    ring = mod.RingBuffer(4)
    assert ring.push([1.0, 2.0, 3.0, 4.0]) == 4
    assert ring.push([5.0]) == 0          # 满了，新采样被丢弃
    assert ring.dropped == 1
    assert ring.pop(4) == [1.0, 2.0, 3.0, 4.0]  # 未读数据完好


def test_ring_buffer_concurrent_producer_consumer_no_loss():
    ring = mod.RingBuffer(64)
    total = 4000
    consumed = []

    def producer():
        for i in range(total):
            # 阻塞式背压：消费者回收槽位前等待，而不是覆盖/丢弃
            ring.push([float(i)], block=True)

    def consumer():
        while len(consumed) < total:
            consumed.extend(ring.pop(64, block=True))

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start(); t2.start()
    t1.join(timeout=15); t2.join(timeout=15)
    assert not t1.is_alive() and not t2.is_alive()
    assert len(consumed) == total
    assert consumed == [float(i) for i in range(total)]
    assert ring.dropped == 0


# ---------------------------------------------------------------------------
# 修复 5：暂停恢复清空
# ---------------------------------------------------------------------------

def test_buffer_clear_resets_and_invalidates_old_generation():
    ring = mod.RingBuffer(16)
    ring.push([1.0, 2.0, 3.0, 4.0])
    assert ring.readable() == 4
    report = mod.buffer_clear(ring)
    assert report["ok"] is True
    assert ring.readable() == 0
    # 旧代读者拿不到任何旧片段
    assert ring.pop(4, generation=report["generation"] - 1) == []
    # 新代写入正常
    assert ring.push([9.0], generation=report["generation"]) == 1
    assert ring.pop(1, generation=report["generation"]) == [9.0]


def test_buffer_clear_bumps_generation_each_time():
    ring = mod.RingBuffer(8)
    first = mod.buffer_clear(ring)["generation"]
    second = mod.buffer_clear(ring)["generation"]
    assert second == first + 1
    assert ring.clears == 2


# ---------------------------------------------------------------------------
# 修复 6：时间轴对齐与混音
# ---------------------------------------------------------------------------

def test_timeline_align_subtracts_compensation_and_converts_rate():
    # 设备位置 2000，补偿 1024，48k -> 24k
    assert mod.timeline_align(
        2000, compensation=1024, sample_rate=48000,
        output_rate=24000) == round(976 / 48000 * 24000)


def test_timeline_align_segment_uses_its_own_sample_rate():
    seg = {"device_start": 640, "sample_rate": 24000,
           "compensation": 512, "samples": [0.1]}
    placed = mod.timeline_align(seg, sample_rate=48000, output_rate=24000)
    # (640-512)/24000 秒，映射回 24k -> 128
    assert placed["timeline_start"] == 128
    assert placed["timeline_start_seconds"] == pytest.approx(128 / 24000)


def test_timeline_align_prefers_explicit_anchor_without_double_compensation():
    seg = {"timeline_start_seconds": 0.5, "sample_rate": 24000,
           "samples": [0.1]}
    placed = mod.timeline_align(seg, compensation=999,
                                sample_rate=48000, output_rate=24000)
    assert placed["timeline_start"] == 12000  # 0.5s * 24k，补偿不再二次扣减


def test_mix_overlay_and_replace_modes():
    backing = [0.5, 0.5, 0.5, 0.5]
    out = mod.mix(backing, [
        {"offset": 1, "samples": [0.5], "mode": "overlay"},
        {"offset": 2, "samples": [-0.25], "mode": "replace"},
    ])
    assert out[0] == 0.5
    assert out[1] == pytest.approx(1.0)      # 0.5+0.5 钳制
    assert out[2] == -0.25                   # 替换
    assert out[3] == 0.5


def test_mix_integer_length_returns_silence():
    assert mod.mix(3) == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 端到端编排
# ---------------------------------------------------------------------------

def test_record_end_toend_all_six_fixes_together():
    out = mod.record(mod.AudioConfig(seed=7), blocks=16)
    assert out["ok"] is True
    metrics = out["metrics"]
    # 修复 4：无丢帧；修复 5：恰好一次清空；修复 3：恰好一次切换
    assert metrics["dropped_frames"] == 0
    assert metrics["buffer_clears"] == 1
    assert metrics["device_switches"] == 1
    # 测量值与真实物理延迟零误差
    assert metrics["latency_error_samples"] == 0
    # 块4(48k) + 块5..7(24k)，暂停块不落盘、区间外块不落盘
    rates = [s["sample_rate"] for s in out["segments"]]
    assert rates == [48000, 24000, 24000, 24000]
    # 时间轴上首尾相接，无重叠无空洞
    starts = [s["timeline_start"] for s in out["segments"]]
    lengths = [
        len(mod._resample_linear(s["samples"], s["sample_rate"], 24000))
        for s in out["segments"]
    ]
    for i in range(1, len(starts)):
        assert starts[i] == starts[i - 1] + lengths[i - 1]
    # 首段落在 punch 起点
    assert starts[0] == round(
        out["punch"]["start"] / 48000 * out["sample_rate"])
    # 混音长度覆盖整条主时间轴
    assert len(out["mix"]) == round(
        out["total_seconds"] * out["sample_rate"])
    # 审计轨迹包含全部关键事件
    events = {e["event"] for e in out["audit"]}
    assert {"session.start", "punch.range", "pause", "resume.clear",
            "device.switch", "delay.rebuilt", "mix.done"} <= events


def test_record_small_session_has_no_punch_segments():
    out = mod.record(1)
    assert out["ok"] is True
    assert out["segments"] == []
    assert len(out["mix"]) == mod.DEFAULT_BLOCK_SIZE
    assert out["metrics"]["device_switches"] == 0


def test_record_resume_clear_uses_new_generation():
    out = mod.record(mod.AudioConfig(seed=4), blocks=16)
    clear_events = [e for e in out["audit"] if e["event"] == "resume.clear"]
    assert len(clear_events) == 1
    new_generation = clear_events[0]["generation"]
    assert new_generation == 1
    # 所有落盘段都属于清空后的新代
    assert all(s["generation"] == new_generation
               for s in out["segments"])


def test_determinism_same_seed_identical_output():
    report = mod.determinism(seed=3, blocks=16)
    assert report["ok"] is True
    assert report["mismatches"] == []


def test_validation_accepts_good_config_and_rejects_bad():
    parsed = mod.validation({"sample_rate": 44100, "block_size": 128})
    assert parsed["sample_rate"] == 44100
    assert parsed["block_size"] == 128
    for bad in ({"sample_rate": 0}, {"block_size": -1},
                {"ring_capacity": 1}, {"unknown_field": 1}):
        with pytest.raises(ValueError):
            mod.validation(bad)


def test_audit_and_metrics_snapshots_are_copies():
    out = mod.record(mod.AudioConfig(seed=2), blocks=16)
    # record 结果字典内带完整审计轨迹与指标快照
    assert isinstance(out["audit"], list) and out["audit"]
    assert out["metrics"]["blocks_processed"] == 16
    # 真正的 Session 路径：快照与轨迹是副本，修改不污染会话内部
    device = mod.SimulatedDevice(mod.AudioConfig(seed=1), seed=1)
    session = mod.Session(
        mod.AudioConfig(seed=1), device,
        mod.measure_latency(device, 48000))
    session.audit.add("synthetic.event")
    entries = mod.audit(session)
    entries.append({"tampered": True})
    assert mod.audit(session)[-1]["event"] == "synthetic.event"
    snapshot = mod.metrics(session)
    snapshot["dropped_frames"] = 999
    assert mod.metrics(session)["dropped_frames"] == 0
    # 无会话的脚手架路径返回零值指标与初始轨迹
    assert mod.metrics()["blocks_processed"] == 0
    assert mod.audit()[0]["event"] == "audit.created"
