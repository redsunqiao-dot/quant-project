"""验证门 decide_status 单测（无行情）。"""

from src.research.factor_validate import decide_status


def test_decide_status_active():
    assert (
        decide_status({"ic_mean": 0.03, "mono_mean": 0.1}, min_abs_ic=0.02)
        == "active"
    )


def test_decide_status_inactive_low_ic():
    assert (
        decide_status({"ic_mean": 0.01, "mono_mean": 0.2}, min_abs_ic=0.02)
        == "inactive"
    )


def test_decide_status_inactive_bad_mono():
    assert (
        decide_status({"ic_mean": 0.05, "mono_mean": -0.1}, min_abs_ic=0.02)
        == "inactive"
    )
