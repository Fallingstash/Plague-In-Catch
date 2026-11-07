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
    command = [sys.executable, "-u", str(ROOT / script)]
    started = datetime.now(timezone.utc).isoformat()
    print(f"\n===== START {name}: {' '.join(command)} =====", flush=True)
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
    result = {
        "step": name,
        "script": script,
        "status": status,
        "detail": detail,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "log": str(log_path),
    }
    print(f"===== END {name}: {status}; log={log_path} =====", flush=True)
    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    steps = [
        ("exp29crossfit", "experiments/exp29crossfit.py"),
        ("exp30stacking", "experiments/exp30stacking.py"),
    ]
    results = [run_step(name, script) for name, script in steps]

    tuning_path = RESULTS_DIR / "exp31tuning.json"
    tuning_oof = RESULTS_DIR / "exp31oof_raw_si.npy"
    if tuning_path.is_file() and tuning_oof.is_file():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        results.append(
            {
                "step": "exp31tuning",
                "script": "experiments/exp31tuning.py",
                "status": "success",
                "detail": "Reused completed 30-trial tuning report and OOF predictions.",
                "started_at": None,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "log": str(RESULTS_DIR / "exp31tuning.log"),
                "best_mse_raw": tuning.get("best_mse_raw"),
            }
        )
        print(
            f"\nReusing completed exp31 tuning: MSE={tuning.get('best_mse_raw')}; "
            f"log={RESULTS_DIR / 'exp31tuning.log'}",
            flush=True,
        )
    else:
        results.append(run_step("exp31tuning", "experiments/exp31tuning.py"))

    if tuning_path.is_file():
        try:
            tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
            if tuning.get("improved_over_baseline"):
                print(
                    "\nTuned LightGBM improved baseline; refitting stack OOF with "
                    "the selected tuned predictions.",
                    flush=True,
                )
                results.append(run_step("exp30stacking_refit_tuned", "experiments/exp30stacking.py"))
        except Exception as error:
            detail = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
            print(f"Tuning selection check failed: {detail}", flush=True)
            results.append(
                {
                    "step": "exp30stacking_refit_tuned",
                    "status": "failed",
                    "detail": detail,
                }
            )

    results.append(run_step("makesubmissionfinal", "makesubmissionfinal.py"))
    report = {
        "task": "TASK 30 global upgrade and stacking pipeline",
        "started_at": results[0]["started_at"] if results else None,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "steps": results,
        "all_steps_succeeded": all(item["status"] == "success" for item in results),
        "outputs": [
            str(RESULTS_DIR / "exp29crossfit.json"),
            str(RESULTS_DIR / "exp30stacking.json"),
            str(RESULTS_DIR / "exp31tuning.json"),
            str(RESULTS_DIR / "makesubmissionfinal.json"),
            str(ROOT / "submissionmlfinal.csv"),
        ],
    }
    report_path = RESULTS_DIR / "runallupgrade.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPipeline report: {report_path}", flush=True)
    print(f"All steps succeeded: {report['all_steps_succeeded']}", flush=True)


if __name__ == "__main__":
    main()
