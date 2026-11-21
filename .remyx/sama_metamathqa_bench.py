#!/usr/bin/env python
"""Validate benchmark wrapper: run the SaMA method_comparison cell and emit ONLY clean metrics on stdout.

run.py is noisy (tqdm, prints) which buried the metrics line and defeated the criteria parser. Here we
CAPTURE run.py's output (so stdout stays clean), then emit the metrics as a single JSON line + the
METRICS_JSON marker + a metrics.json file — whichever the reader prefers.
"""
import glob, json, os, subprocess, sys

MC = os.path.join("method_comparison", "MetaMathQA")
EXP = os.path.join("experiments", "sama", "llama-3.2-3B-8x8")


def _newest_results():
    pats = [os.path.join(MC, "results", "sama--*.json"),
            os.path.join(MC, "temporary_results", "sama--*.json")]
    files = [f for p in pats for f in glob.glob(p)]
    return max(files, key=os.path.getmtime) if files else None


def main():
    # capture run.py output so it does NOT pollute our metrics stdout
    proc = subprocess.run([sys.executable, "run.py", "-v", EXP], cwd=MC,
                          capture_output=True, text=True)
    sys.stderr.write(proc.stdout[-4000:])          # training log -> stderr (visible, not parsed)
    sys.stderr.write(proc.stderr[-4000:])
    if proc.returncode != 0:
        sys.stderr.write(f"\nrun.py exited {proc.returncode}\n"); sys.exit(proc.returncode)
    rf = _newest_results()
    if not rf:
        sys.stderr.write("no SaMA results json found\n"); sys.exit(1)
    d = json.load(open(rf)); ti = d.get("train_info", {}); last = (ti.get("metrics") or [{}])[-1]
    metrics = {
        "gsm8k_test_accuracy": last.get("test accuracy"),
        "trainable_params_m": round((ti.get("num_trainable_params") or 0) / 1e6, 3),
        "train_time_s": round(ti.get("train_time") or 0, 1),
        "peak_mem_gb": round((ti.get("accelerator_memory_max") or 0) / 1e9, 2),
    }
    with open(os.path.join(os.path.dirname(__file__), "metrics.json"), "w") as f:
        json.dump(metrics, f)
    # clean stdout: the only things printed are the metrics
    print(json.dumps(metrics))
    print("METRICS_JSON=" + json.dumps(metrics))


if __name__ == "__main__":
    main()
