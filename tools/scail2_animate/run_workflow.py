#!/usr/bin/env python3
"""Submit one prepared image/FBX SCAIL-2 graph and wait for completion."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import make_workflow
import workflow_builder
from _env import INPUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--seed", type=int, default=260724)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--subject", default="person")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    server = args.server.rstrip("/")

    run_dir, report = make_workflow.load_run(args.run_dir)
    graph = make_workflow.workflow(
        run_dir.relative_to(INPUT_ROOT).as_posix(),
        json.dumps(report, sort_keys=True, separators=(",", ":")),
        f"jakkanna_animate/{report['run_id']}",
        args.prompt,
        args.subject,
        args.seed,
        report["settings"]["width"],
        report["settings"]["height"],
    )
    prompt = workflow_builder.to_api(server, graph)

    request = urllib.request.Request(
        f"{server}/prompt",
        json.dumps({"prompt": prompt}).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            submitted = json.load(response)
    except urllib.error.HTTPError as error:
        print("submit failed:", error.read().decode()[:4000])
        sys.exit(1)
    if submitted.get("node_errors"):
        print("node errors:", json.dumps(submitted["node_errors"], indent=2)[:4000])
        sys.exit(1)

    prompt_id = submitted["prompt_id"]
    print(f"queued {prompt_id}", flush=True)
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > args.timeout:
            print(f"timed out after {args.timeout:.0f}s")
            sys.exit(1)
        time.sleep(args.poll_seconds)
        with urllib.request.urlopen(f"{server}/history/{prompt_id}") as response:
            history = json.load(response)
        if prompt_id not in history:
            print(f"  running... {elapsed:6.0f}s", flush=True)
            continue

        entry = history[prompt_id]
        status = entry.get("status", {})
        print(f"finished in {elapsed:.0f}s: {status.get('status_str')}")
        if status.get("status_str") != "success":
            for message in status.get("messages", []):
                print("  ", json.dumps(message)[:2000])
            sys.exit(1)
        for node_id, output in entry.get("outputs", {}).items():
            node_type = prompt[node_id]["class_type"]
            values = {
                key: len(value) if isinstance(value, list) else value
                for key, value in output.items()
            }
            print(f"  node {node_id} ({node_type}): {values}")
        return


if __name__ == "__main__":
    main()
