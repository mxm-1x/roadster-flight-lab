"""Local viewer and read-only experiment/replay API."""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import mimetypes
import re
import shutil
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pandas as pd
import plotly
import torch

from rocket_landing.config import VARIANTS, VARIANT_LABELS, resolve_output_dir
from rocket_landing.replay import flight_figure, load_training_episode, replay_episode
from rocket_landing.visualization_data import evaluation_outcomes, read_training_episodes, read_training_progress

WEB_ROOT = Path(__file__).resolve().parent / "web_assets"
PLOTLY_JS = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
SAFE_SESSION = re.compile(r"^[A-Za-z0-9_-]+$")


def _records(data: pd.DataFrame) -> list[dict]:
    clean = data.replace([np.inf, -np.inf], np.nan)
    return clean.astype(object).where(pd.notna(clean), None).to_dict("records")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def bootstrap_showcase(source: Path, outputs: Path) -> int:
    """Seed a new runtime directory without overwriting existing experiment runs."""
    source, outputs = source.resolve(), outputs.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Showcase directory not found: {source}")
    if source == outputs:
        return 0
    copied = 0
    for model in sorted(source.glob("*/seed_*/model.zip")):
        run = model.parent
        destination = outputs / run.relative_to(source)
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".showcase-", dir=destination.parent))
        try:
            shutil.copytree(run, temporary, dirs_exist_ok=True)
            temporary.rename(destination)
            copied += 1
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return copied


class ExperimentData:
    def __init__(self, outputs: Path):
        self.outputs = outputs.resolve()

    def run_dir(self, variant: str, seed: int) -> Path:
        if variant not in VARIANTS or not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2_147_483_647:
            raise ValueError("Unknown policy or seed.")
        return self.outputs / variant / f"seed_{seed}"

    def runs(self) -> list[dict]:
        result = []
        for variant in VARIANTS:
            for run in sorted((self.outputs / variant).glob("seed_*")):
                try:
                    seed = int(run.name.removeprefix("seed_"))
                    self.run_dir(variant, seed)
                except ValueError:
                    continue
                model = run / "model.zip"
                if not model.is_file() and not (run / "training_monitor.csv").is_file():
                    continue
                status = _read_json(run / "training_status.json").get("state")
                result.append({"variant": variant, "label": VARIANT_LABELS[variant], "seed": seed,
                               "has_model": model.is_file(), "status": status})
        return result

    def detail(self, variant: str, seed: int) -> dict:
        run = self.run_dir(variant, seed)
        if not run.is_dir():
            raise FileNotFoundError("Policy run not found.")
        model = run / "model.zip"
        summary_file = run / "evaluation_summary.json"
        episode_file = run / "evaluation_episodes.csv"
        evaluation = []
        summary = {}
        saved_summary = _read_json(summary_file)
        digest = saved_summary.get("model_sha256")
        current = (digest == hashlib.sha256(model.read_bytes()).hexdigest()) if digest and model.is_file() else (model.is_file() and summary_file.is_file() and episode_file.is_file() and min(summary_file.stat().st_mtime, episode_file.stat().st_mtime) >= model.stat().st_mtime)
        if model.is_file() and summary_file.is_file() and episode_file.is_file() and current:
            summary = saved_summary
            summary.pop("model_path", None)
            try:
                frame = pd.read_csv(episode_file)
                frame["outcome"] = evaluation_outcomes(frame)
                evaluation = _records(frame)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, KeyError):
                summary = {}
        checkpoints = []
        for path in (run / "checkpoints").glob("ppo_lunar_lander_*_steps.zip"):
            try:
                checkpoints.append(int(path.stem.rsplit("_", 2)[1]))
            except ValueError:
                continue
        sessions = []
        for group in sorted((run / "training_trajectories").glob("*"), reverse=True):
            if not group.is_dir() or not SAFE_SESSION.fullmatch(group.name):
                continue
            index_file = group / "episodes.csv"
            try:
                index = pd.read_csv(index_file) if index_file.is_file() else pd.DataFrame()
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                index = pd.DataFrame()
            outcomes = dict(zip(index["episode"], index["outcome"])) if {"episode", "outcome"}.issubset(index.columns) else {}
            recordings = []
            for path in sorted(group.glob("episode_*.npz")):
                try:
                    number = int(path.stem.split("_")[-1])
                except ValueError:
                    continue
                recordings.append({"episode": number, "outcome": outcomes.get(number, "Unknown")})
            if recordings:
                sessions.append({"id": group.name, "episodes": recordings})
        return {"variant": variant, "seed": seed, "has_model": model.is_file(),
                "status": _read_json(run / "training_status.json"),
                "training": _records(read_training_episodes(run)),
                "progress": _records(read_training_progress(run)),
                "evaluation": evaluation, "summary": summary,
                "checkpoints": sorted(checkpoints), "sessions": sessions}

    def replay(self, payload: dict) -> dict:
        variant, seed = payload.get("variant"), payload.get("seed")
        run = self.run_dir(variant, seed)
        mode = payload.get("mode")
        if mode == "evaluation":
            episode = payload.get("episode")
            if not isinstance(episode, int) or isinstance(episode, bool):
                raise ValueError("Choose an evaluation episode.")
            detail = self.detail(variant, seed)
            row = next((row for row in detail["evaluation"] if row["episode"] == episode), None)
            if row is None or not detail["has_model"]:
                raise ValueError("Evaluation episode is not available for this model.")
            flight, pad = replay_episode(run / "model.zip", variant, int(row["eval_seed"]))
        elif mode == "checkpoint":
            step, episode_seed = payload.get("checkpoint"), payload.get("episode_seed")
            if (not isinstance(step, int) or isinstance(step, bool) or step not in self.detail(variant, seed)["checkpoints"]
                    or not isinstance(episode_seed, int) or isinstance(episode_seed, bool) or not 0 <= episode_seed <= 2_147_483_647):
                raise ValueError("Choose a valid checkpoint and replay seed.")
            flight, pad = replay_episode(run / "checkpoints" / f"ppo_lunar_lander_{step}_steps.zip", variant, episode_seed)
        elif mode == "recorded":
            session, episode = payload.get("session"), payload.get("episode")
            if (not isinstance(session, str) or not SAFE_SESSION.fullmatch(session)
                    or not isinstance(episode, int) or isinstance(episode, bool) or episode < 1):
                raise ValueError("Choose a valid recorded training episode.")
            folder = (run / "training_trajectories" / session).resolve()
            if folder.parent != (run / "training_trajectories").resolve():
                raise ValueError("Invalid training session.")
            path = folder / f"episode_{episode:07d}.npz"
            if not path.is_file():
                raise FileNotFoundError("Recorded training episode not found.")
            flight, pad = load_training_episode(path)
        else:
            raise ValueError("Choose an available flight source.")
        figure_obj = flight_figure(flight, pad)
        figure_obj.update_layout(margin=dict(l=20, r=20, t=15, b=60), template=None, paper_bgcolor="#111111", plot_bgcolor="#111111", font=dict(color="#d4d4d4"),
                                 scene=dict(bgcolor="#111111", aspectmode="manual", aspectratio=dict(x=1.4, y=0.35, z=1.2),
                                            camera=dict(eye=dict(x=1.1, y=-1.6, z=0.8)),
                                            xaxis=dict(gridcolor="#333333", zerolinecolor="#555555", color="#999999", showbackground=False),
                                            yaxis=dict(gridcolor="#333333", zerolinecolor="#555555", color="#999999", showbackground=False),
                                            zaxis=dict(gridcolor="#333333", zerolinecolor="#555555", color="#999999", showbackground=False)))
        for menu in figure_obj.layout.updatemenus:
            menu.update(bgcolor="#222222", bordercolor="#444444", font=dict(color="#dddddd"))
        for slider in figure_obj.layout.sliders:
            slider.update(bgcolor="#555555", activebgcolor="#eeeeee", bordercolor="#444444", font=dict(color="#999999"))
        figure = json.loads(figure_obj.to_json())
        return {"figure": figure, "flight": _records(flight), "outcome": flight.attrs.get("outcome", "Unknown"),
                "native_return": float(flight["native_total"].iloc[-1]),
                "shaped_return": float(flight["shaped_total"].iloc[-1]), "steps": len(flight) - 1}


def make_handler(outputs: Path):
    data = ExperimentData(outputs)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, status=HTTPStatus.OK):
            content = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/health":
                return self._json({"status": "ok"})
            if path == "/api/runs":
                return self._json({"runs": data.runs()})
            match = re.fullmatch(r"/api/runs/([a-z_]+)/(\d+)", path)
            if match:
                try:
                    return self._json(data.detail(match.group(1), int(match.group(2))))
                except (ValueError, FileNotFoundError) as exc:
                    return self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            if path == "/plotly.min.js":
                asset = PLOTLY_JS
            else:
                candidate = (WEB_ROOT / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
                asset = candidate if candidate.is_relative_to(WEB_ROOT.resolve()) else None
            if asset is None or not asset.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content = asset.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(asset.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path != "/api/replay":
                return self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            try:
                origin = self.headers.get("Origin")
                if origin and (urlsplit(origin).netloc != self.headers.get("Host") or urlsplit(origin).scheme not in ("http", "https")):
                    return self._json({"error": "Origin not allowed."}, HTTPStatus.FORBIDDEN)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_384:
                    raise ValueError("Invalid request size.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                return self._json(data.replay(payload))
            except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception:
                return self._json({"error": "Could not replay this flight. Check the saved model and try again."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run the local saved-results website.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--output-dir", type=Path, default=resolve_output_dir())
    parser.add_argument("--showcase-dir", type=Path, help="Copy bundled models into a fresh runtime directory at startup.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    outputs = resolve_output_dir(args.output_dir)
    if args.showcase_dir:
        count = bootstrap_showcase(args.showcase_dir, outputs)
        print(f"Initialized {count} bundled experiment runs", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(outputs))
    print(f"Rocket landing website: http://localhost:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
