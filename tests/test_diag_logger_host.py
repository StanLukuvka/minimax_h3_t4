from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

from src.minimax_h3_t4.runtime import diag_logger


def test_log_host_output_reports_rss() -> None:
    """log_host must emit a HOST line carrying the current process RSS."""
    with mock.patch.dict(os.environ, {"H3_T4_DIAG": "1"}):
        diag_logger._DIAG_ENABLED = True
        with mock.patch(
            "src.minimax_h3_t4.runtime.diag_logger._get_host_rss_bytes",
            return_value=3 * 1024**3,
        ):
            captured: list[str] = []
            with mock.patch(
                "builtins.print", side_effect=lambda *a, **k: captured.append(" ".join(a))
            ):
                diag_logger.log_host("load_unet_start")
    assert any("HOST" in line and "rss=" in line and "load_unet_start" in line for line in captured)
    assert any("3.00GiB" in line for line in captured)


def test_log_host_respects_disable_env() -> None:
    """With H3_T4_DIAG disabled, log_host must not print anything."""
    diag_logger._DIAG_ENABLED = False
    with mock.patch("builtins.print") as print_mock:
        diag_logger.log_host("should_be_silent")
    print_mock.assert_not_called()


def test_get_host_rss_uses_psutil_when_present() -> None:
    """When psutil is importable, RSS comes from psutil.Process.memory_info().rss."""
    fake_rss = 1234567

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def memory_info(self):
            return type("Mem", (), {"rss": fake_rss})()

    class FakePsutil:
        Process = FakeProcess

    # Clear cached module if present so we get fresh import
    sys.modules.pop("psutil", None)
    with mock.patch.dict("sys.modules", {"psutil": FakePsutil()}):
        # Re-import the module to pick up the patched psutil
        original_module = sys.modules.get("src.minimax_h3_t4.runtime.diag_logger")
        try:
            del sys.modules["src.minimax_h3_t4.runtime.diag_logger"]
            del sys.modules["src.minimax_h3_t4.runtime"]
            from src.minimax_h3_t4.runtime import diag_logger as fresh_diag

            assert fresh_diag._get_host_rss_bytes() == fake_rss
        finally:
            if original_module is not None:
                sys.modules["src.minimax_h3_t4.runtime.diag_logger"] = original_module
                sys.modules["src.minimax_h3_t4.runtime"] = __import__("src.minimax_h3_t4.runtime")


def test_get_host_rss_returns_negative_when_unavailable() -> None:
    """If neither psutil nor /proc is available, return -1 sentinel."""
    # Patch at function level to bypass the closed-over Path reference
    with mock.patch(
        "src.minimax_h3_t4.runtime.diag_logger.Path",
        spec=Path,
    ) as fake_path_cls:
        fake_instance = mock.MagicMock()
        fake_instance.is_file.return_value = False
        fake_path_cls.return_value = fake_instance
        with mock.patch.dict("sys.modules", {"psutil": None}):
            assert diag_logger._get_host_rss_bytes() == -1


def test_log_host_reports_unavailable_when_rss_negative() -> None:
    """A -1 RSS should print 'rss=unavailable' rather than a bogus size."""
    diag_logger._DIAG_ENABLED = True
    with mock.patch(
        "src.minimax_h3_t4.runtime.diag_logger._get_host_rss_bytes",
        return_value=-1,
    ):
        captured: list[str] = []
        with mock.patch(
            "builtins.print", side_effect=lambda *a, **k: captured.append(" ".join(a))
        ):
            diag_logger.log_host("load_unet_start")
    assert any("rss=unavailable" in line for line in captured)


def test_write_summary_remains_callable() -> None:
    """Sanity: write_summary is importable and accepts a path + mapping."""
    assert callable(diag_logger.write_summary)
