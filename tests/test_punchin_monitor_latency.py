import pytest
import media.punchin_monitor_latency as mod

CASES = [
    ("measure_latency", (1,)),
    ("sample_align", (1,)),
    ("device_recalc", (1,)),
    ("ring_fence", (1,)),
    ("buffer_clear", (1,)),
    ("timeline_align", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("monitor_path", (1,)),
    ("metrics", (1,)),
    ("punch_range", (1,)),
    ("mix", (1,)),
    ("record", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
