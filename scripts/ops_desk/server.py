#!/usr/bin/env python3
"""
本机作战台：日更 / 回测 / 因子体检 / 出图。

启动:
  .venv/bin/python scripts/ops_desk/server.py
  浏览器 http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ops_desk.data_apis import (  # noqa: E402
    OUT_DIR,
    PLOTS_DIR,
    build_builtin_charts,
    build_factors_snapshot,
    build_ops_snapshot,
    list_course_viz,
    load_backtest_snapshot,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
_REFRESH_LOCK = threading.Lock()
_REFRESH_STATE = {"running": False, "message": "", "ok": None}
_SIGNAL_LOCK = threading.Lock()
_SIGNAL_STATE = {
    "running": False,
    "track": "",
    "message": "",
    "ok": None,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"[ops] {args[0]}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code: int = 200) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(code, raw, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send(
                200,
                (STATIC_DIR / "index.html").read_bytes(),
                "text/html; charset=utf-8",
            )
            return
        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            target = (STATIC_DIR / rel).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            ctype = "text/css; charset=utf-8"
            if target.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            self._send(200, target.read_bytes(), ctype)
            return
        if path == "/api/ops":
            self._send_json(build_ops_snapshot())
            return
        if path == "/api/signal/status":
            self._send_json(dict(_SIGNAL_STATE))
            return
        if path == "/api/backtest":
            self._send_json(load_backtest_snapshot())
            return
        if path == "/api/backtest/status":
            self._send_json(dict(_REFRESH_STATE))
            return
        if path == "/api/factors":
            self._send_json(build_factors_snapshot())
            return
        if path == "/api/charts":
            self._send_json(build_builtin_charts())
            return
        if path == "/api/viz/list":
            self._send_json({"scripts": list_course_viz()})
            return
        if path.startswith("/api/plots/"):
            name = unquote(path[len("/api/plots/") :])
            target = (PLOTS_DIR / name).resolve()
            if not str(target).startswith(str(PLOTS_DIR.resolve())) or not target.is_file():
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            ctype = "image/png" if target.suffix == ".png" else "image/svg+xml"
            self._send(200, target.read_bytes(), ctype)
            return
        if path.startswith("/api/outputs_file/"):
            rel = unquote(path[len("/api/outputs_file/") :])
            target = (ROOT / "outputs" / rel).resolve()
            out_root = (ROOT / "outputs").resolve()
            if not str(target).startswith(str(out_root)) or not target.is_file():
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            ctype = "image/png" if target.suffix == ".png" else "application/octet-stream"
            self._send(200, target.read_bytes(), ctype)
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            body = {}

        if path == "/api/backtest/refresh":
            self._start_backtest_refresh(body)
            return
        if path == "/api/signal/refresh":
            self._start_signal_refresh(body)
            return
        if path == "/api/viz/run":
            self._run_viz(body)
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def _start_backtest_refresh(self, body: dict) -> None:
        track = str(body.get("track") or "main").lower()
        if track not in ("main", "st", "both"):
            track = "main"
        with _REFRESH_LOCK:
            if _REFRESH_STATE["running"]:
                self._send_json({"ok": False, "message": "回测刷新已在进行中"}, 409)
                return
            _REFRESH_STATE.update(
                {"running": True, "message": f"回测运行中（{track}）…", "ok": None}
            )

        def worker() -> None:
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "ops_desk" / "refresh_backtest.py"),
                "--track",
                track,
            ]
            if body.get("strategy"):
                cmd += ["--strategy", str(body["strategy"])]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                )
                if proc.returncode == 0:
                    msg, ok = f"回测缓存已更新（{track}）", True
                else:
                    msg = (proc.stderr or proc.stdout or "失败")[-500:]
                    ok = False
            except Exception as exc:
                msg, ok = str(exc), False
            with _REFRESH_LOCK:
                _REFRESH_STATE.update(
                    {"running": False, "message": msg, "ok": ok}
                )

        threading.Thread(target=worker, daemon=True).start()
        self._send_json({"ok": True, "message": f"已开始刷新回测（{track}）"})

    def _start_signal_refresh(self, body: dict) -> None:
        """重算 live 清单 + paper 账本。run_pipeline=true 时先补合成分。"""
        track = str(body.get("track") or "main").lower()
        run_pipeline = bool(body.get("run_pipeline"))
        if track not in ("main", "st", "both"):
            self._send_json(
                {"ok": False, "message": "track 应为 main / st / both"}, 400
            )
            return
        with _SIGNAL_LOCK:
            if _SIGNAL_STATE["running"]:
                self._send_json(
                    {
                        "ok": False,
                        "message": f"信号刷新进行中: {_SIGNAL_STATE['track']}",
                    },
                    409,
                )
                return
            mode = "含管线" if run_pipeline else "仅信号"
            _SIGNAL_STATE.update(
                {
                    "running": True,
                    "track": track,
                    "message": f"正在刷新{mode}（{track}）…",
                    "ok": None,
                }
            )

        def worker() -> None:
            jobs = []
            if track in ("main", "both"):
                if run_pipeline:
                    jobs.append(
                        (
                            "主轨管线",
                            [
                                sys.executable,
                                str(ROOT / "main.py"),
                                "--pipeline",
                            ],
                        )
                    )
                jobs.append(
                    (
                        "主轨信号",
                        [
                            sys.executable,
                            str(ROOT / "scripts" / "run_paper_pipeline.py"),
                            "--skip-pipeline",
                        ],
                    )
                )
            if track in ("st", "both"):
                st_cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_paper_st_pipeline.py"),
                ]
                if not run_pipeline:
                    st_cmd.append("--skip-pipeline")
                jobs.append(("短线", st_cmd))
            ok_all = True
            notes = []
            try:
                for label, cmd in jobs:
                    with _SIGNAL_LOCK:
                        _SIGNAL_STATE["message"] = f"正在跑{label}…"
                    proc = subprocess.run(
                        cmd,
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode == 0:
                        notes.append(f"{label}完成")
                    else:
                        ok_all = False
                        err = (proc.stderr or proc.stdout or "失败")[-300:]
                        notes.append(f"{label}失败: {err}")
                msg = "；".join(notes)
            except Exception as exc:
                ok_all = False
                msg = str(exc)
            with _SIGNAL_LOCK:
                _SIGNAL_STATE.update(
                    {
                        "running": False,
                        "track": track,
                        "message": msg,
                        "ok": ok_all,
                    }
                )

        threading.Thread(target=worker, daemon=True).start()
        self._send_json(
            {
                "ok": True,
                "message": f"已开始刷新信号（{track}{' 含管线' if run_pipeline else ''}）",
            }
        )

    def _run_viz(self, body: dict) -> None:
        script_id = str(body.get("id") or "")
        viz = ROOT / "scripts" / "course_viz" / f"{script_id}.py"
        if not viz.exists():
            self._send_json({"ok": False, "message": f"脚本不存在: {script_id}"}, 404)
            return
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        # 多数 course_viz 自带输出路径；再额外尝试把 cwd 下 png 拷到 ops_desk/plots
        before = {p.name for p in ROOT.joinpath("outputs").rglob("*.png")}
        proc = subprocess.run(
            [sys.executable, str(viz)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        after = {p for p in ROOT.joinpath("outputs").rglob("*.png")}
        new_files = []
        for p in after:
            if p.name not in before or p.stat().st_mtime > (proc.returncode):
                # 复制新图到 plots
                try:
                    dest = PLOTS_DIR / f"{script_id}_{p.name}"
                    dest.write_bytes(p.read_bytes())
                    new_files.append(dest.name)
                except Exception:
                    pass
        self._send_json(
            {
                "ok": proc.returncode == 0,
                "message": (proc.stdout or "")[-800:]
                or (proc.stderr or "")[-800:]
                or ("完成" if proc.returncode == 0 else "失败"),
                "images": new_files,
                "returncode": proc.returncode,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="本机作战台网页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if not (STATIC_DIR / "index.html").exists():
        raise SystemExit(f"缺少页面: {STATIC_DIR / 'index.html'}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"作战台: http://{args.host}:{args.port}")
    print("Ctrl+C 退出")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
