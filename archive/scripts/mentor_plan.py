import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "experiments" / "results"


def run_step(name, script):
    log_path = RESULTS_DIR / f"{name}.log"
    started = datetime.now(timezone.utc).isoformat()
    command = [sys.executable, "-u", str(ROOT / script)]
    print(f"\n===== START {name} =====", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        status = "success" if return_code == 0 else "failed"
        detail = None if return_code == 0 else f"Process exited with code {return_code}"
    except Exception as error:
        status = "failed"
        detail = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(detail)
        print(detail, flush=True)
    return {
        "step": name,
        "script": script,
        "status": status,
        "detail": detail,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "log": str(log_path),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        run_step("exp35outlier", "experiments/exp35outlier.py"),
        run_step("exp36chembl", "experiments/exp36chembl_feature.py"),
    ]

    teammate_path = ROOT / "teammate_submission.csv"
    if teammate_path.is_file():
        results.append(
            {
                "step": "exp37teammate_ensemble",
                "status": "not_run",
                "detail": "TASK requested stopping after priorities 1 and 2.",
                "input_present": True,
            }
        )
        print(
            "teammate_submission.csv exists; priority 3 not run because the request "
            "says to stop after priorities 1 and 2.",
            flush=True,
        )
    else:
        results.append(
            {
                "step": "exp37teammate_ensemble",
                "status": "skipped",
                "detail": "Ждем файл от сокомандника. Пропускаем",
                "input_present": False,
            }
        )
        print("Ждем файл от сокомандника. Пропускаем", flush=True)

    report = {
        "task": "TASK 35 mentor priority plan",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "steps": results,
        "priorities_1_and_2_succeeded": all(
            item["status"] == "success" for item in results[:2]
        ),
    }
    report_path = RESULTS_DIR / "mentor_plan.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved pipeline report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
