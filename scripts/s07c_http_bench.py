"""Stage 7c: HTTP latency benchmark of POST /v1/classify with 16 concurrent users (honest numbers: warm-up discarded, real val length mix,
load generator runs on the SAME machine => it steals CPU; hardware is recorded)."""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
import numpy as np

from bayan import config as C
from bayan.data import load_feedback
from bayan.onnx_utils import hardware_info
from bayan.splits import get_part, load_splits
from bayan.train_utils import save_json

ap = argparse.ArgumentParser()
ap.add_argument("--precision", default="fp32", choices=["fp32", "int8"])
ap.add_argument("--users", type=int, default=16)
ap.add_argument("--seconds", type=int, default=30)
ap.add_argument("--port", type=int, default=8008)
ap.add_argument("--max-batch", type=int, default=16)
ap.add_argument("--max-wait-ms", type=float, default=2.0)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--think-ms-list", default="0", help="comma list of mean think times (ms) per user, one load run each in the same server session. "
                "0 = closed-loop saturation (the literal '16 concurrent users' test); >0 = paced users (exponential think time), i.e. a fixed offered load")
ap.add_argument("--tag", default="")
ap.add_argument("--pin", action="store_true", help="pin the server to CPU 0 and the load generator to CPU 1 (needs >=2 cores)")
ap.add_argument("--model", default="", help="override the served general topic model (name under models/)")
ap.add_argument("--no-route", action="store_true", help="benchmark the single general classifier only (dialect router disabled)")
a = ap.parse_args()

df = load_feedback()
texts = get_part(df, load_splits(), "val").text.tolist()
rng = np.random.RandomState(C.SEED)
mix = [texts[i] for i in rng.permutation(len(texts))[:500]]

env = dict(os.environ, BAYAN_PRECISION=a.precision, BAYAN_MAX_BATCH=str(a.max_batch), BAYAN_MAX_WAIT_MS=str(a.max_wait_ms),
           BAYAN_THREADS=str(a.threads), BAYAN_CPU_ONLY="1", CUDA_VISIBLE_DEVICES="")
if a.model:
    env["BAYAN_TOPIC_MODEL"] = a.model
if a.no_route:
    env["BAYAN_NO_ROUTE"] = "1"
log = open(C.WORK_DIR / "logs" / f"server_{a.precision}{a.tag}.log", "w")
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "bayan.service:app_factory", "--factory", "--port", str(a.port), "--log-level", "warning",
                        "--workers", "1"], env=env, stdout=log, stderr=subprocess.STDOUT,
                       preexec_fn=(lambda: (os.setsid(), os.sched_setaffinity(0, {0}))) if a.pin else os.setsid)
if a.pin:
    os.sched_setaffinity(0, {1})
base = f"http://127.0.0.1:{a.port}"
try:
    t0 = time.time()
    while True:
        try:
            r = httpx.get(base + "/health", timeout=2)
            if r.status_code == 200:
                health = r.json()
                break
        except Exception:  # noqa
            pass
        if srv.poll() is not None:
            print(open(C.WORK_DIR / "logs" / f"server_{a.precision}{a.tag}.log").read()[-1500:])
            sys.exit("server exited during start-up (canary failure?)")
        if time.time() - t0 > 400:
            sys.exit("server start-up timeout")
        time.sleep(2)
    print("server up in", round(time.time() - t0, 1), "s", health)

    async def main(think_ms):
        lat, errs = [], 0
        stop_at = time.perf_counter() + a.seconds
        warm_until = time.perf_counter() + 3          # discard the first 3 s of the run
        async with httpx.AsyncClient(base_url=base, timeout=30, limits=httpx.Limits(max_connections=a.users)) as cl:
            for t in mix[:30]:                          # sequential warm-up (discarded)
                await cl.post("/v1/classify", json={"text": t})

            async def user(uid):
                nonlocal errs
                r_ = np.random.RandomState(uid)
                while time.perf_counter() < stop_at:
                    t = mix[r_.randint(len(mix))]
                    s = time.perf_counter()
                    try:
                        resp = await cl.post("/v1/classify", json={"text": t})
                        ok = resp.status_code == 200
                    except Exception:  # noqa
                        ok = False
                    e = time.perf_counter()
                    if not ok:
                        errs += 1
                    elif s >= warm_until:
                        lat.append((e - s) * 1000)
                    if think_ms > 0:
                        await asyncio.sleep(r_.exponential(think_ms) / 1000)

            await asyncio.gather(*[user(i) for i in range(a.users)])
        return np.array(lat), errs

    runs = []
    for think in [float(x) for x in a.think_ms_list.split(",")]:
        lat, errs = asyncio.run(main(think))
        run = {"mean_think_ms": think, "users": a.users, "seconds": a.seconds, "requests": int(len(lat)), "errors": int(errs),
               "rps": round(len(lat) / max(a.seconds - 3, 1), 1), "p50_ms": round(float(np.percentile(lat, 50)), 2),
               "p95_ms": round(float(np.percentile(lat, 95)), 2), "p99_ms": round(float(np.percentile(lat, 99)), 2), "max_ms": round(float(lat.max()), 2),
               "meets_p99_le_40ms": bool(np.percentile(lat, 99) <= 40)}
        print(json.dumps(run))
        runs.append(run)
    sat = runs[0]            # first run is the headline (think time 0 = closed-loop saturation) so older readers of this JSON keep working
    out = {"precision": a.precision, **{k: v for k, v in sat.items() if k != "mean_think_ms"}, "headline_mean_think_ms": sat["mean_think_ms"],
           "runs": runs, "micro_batch": {"max_batch": a.max_batch, "max_wait_ms": a.max_wait_ms},
           "server_threads": a.threads, "hardware": hardware_info(), "service_health": health,
           "caveat": "load generator and server share the same 2-vCPU Colab VM; the lab CPU may differ"}
    out["model"] = a.model or "serving_config.general"
    out["router_enabled"] = not a.no_route
    save_json(out, C.REPORTS_DIR / f"http_bench_{a.precision}{a.tag}.json")
finally:
    try:
        os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
    except ProcessLookupError:                      # server already exited (e.g. canary failure): nothing to kill
        pass
