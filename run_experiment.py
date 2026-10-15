"""Builds the three implementations and runs the full benchmark under the declared budget.

Usage (from the repository root):  python run_experiment.py

Outputs:
  results/raw_measurements.csv  every run (warm-up, measured and memory runs)
  results/run_status.csv        OK / TIMEOUT / ERROR / SKIPPED for every configuration
  results/environment.txt       tool versions and CPU used for the run
"""
import os
import platform
import queue
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

# Toolchain (adjust these paths on another machine)
GCC = r"C:\msys64\mingw64\bin\gcc.exe"
JDK_BIN = r"C:\Program Files\Microsoft\jdk-21.0.12.101-hotspot\bin"
PYTHON = sys.executable

# gcc needs its own bin directory on PATH to find the assembler and linker
os.environ["PATH"] = os.path.dirname(GCC) + os.pathsep + os.environ["PATH"]

# Declared before running: size sequence, repetitions and budget
SIZES = [10, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
REPETITIONS = 5
WARMUPS = {"c": 2, "java": 3, "python": 1}
RUN_BUDGET_SECONDS = 15 * 60          # maximum duration of a single run
MEMORY_BUDGET_GB = 30                 # largest size planned: 3 x 8 x 3000^2 B = 0.2 GB

COMMANDS = {
    "c": [os.path.join(ROOT, "c", "benchmark.exe")],
    "java": [os.path.join(JDK_BIN, "java.exe"), "-cp", os.path.join(ROOT, "java"), "Benchmark"],
    "python": [PYTHON, os.path.join(ROOT, "python", "benchmark.py")],
}

HEADER = "language,n,dtype,threads,run_type,repetition,time_seconds,memory_bytes,checksum"


def build():
    subprocess.run([GCC, "-O2", "-o", "benchmark.exe", "matmul.c", "benchmark.c"],
                   cwd=os.path.join(ROOT, "c"), check=True)
    subprocess.run([os.path.join(JDK_BIN, "javac.exe"), "MatMul.java", "Benchmark.java"],
                   cwd=os.path.join(ROOT, "java"), check=True)


def tool_output(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip().splitlines()[0]


def save_environment():
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True).stdout.strip()
    lines = [
        f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"os: {platform.platform()}",
        f"cpu: {cpu}",
        f"gcc: {tool_output([GCC, '--version'])}  (flags: -O2)",
        f"java: {tool_output([os.path.join(JDK_BIN, 'java.exe'), '-version'])}",
        f"python: {platform.python_implementation()} {platform.python_version()}",
        f"sizes: {SIZES}",
        f"repetitions: {REPETITIONS}, warm-ups: {WARMUPS}",
        f"budget: {RUN_BUDGET_SECONDS} s per run, {MEMORY_BUDGET_GB} GB memory",
    ]
    with open(os.path.join(RESULTS, "environment.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_configuration(lang, n, raw):
    """Runs one (language, n) configuration; kills it if a single run exceeds the budget."""
    cmd = COMMANDS[lang] + ["run", str(n), str(WARMUPS[lang]), str(REPETITIONS)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=ROOT)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)],
                     daemon=True).start()

    start = last = time.monotonic()
    while True:
        try:
            line = lines.get(timeout=max(0.0, RUN_BUDGET_SECONDS - (time.monotonic() - last)))
        except queue.Empty:
            proc.kill()
            proc.wait()
            return "TIMEOUT", f"a run exceeded {RUN_BUDGET_SECONDS} s", time.monotonic() - start
        if line is None:
            break
        if line.startswith(lang + ","):
            raw.write(line)
            raw.flush()
            last = time.monotonic()
            print("   ", line.strip(), flush=True)

    proc.wait()
    if proc.returncode != 0:
        detail = proc.stderr.read().strip().replace("\n", " ").replace(",", ";")
        return "ERROR", f"exit code {proc.returncode}: {detail}", time.monotonic() - start
    return "OK", "", time.monotonic() - start


def main():
    os.makedirs(RESULTS, exist_ok=True)
    build()
    save_environment()

    with open(os.path.join(RESULTS, "raw_measurements.csv"), "w", newline="") as raw, \
         open(os.path.join(RESULTS, "run_status.csv"), "w", newline="") as status:
        raw.write(HEADER + "\n")
        status.write("language,n,status,detail,wall_seconds\n")

        for lang in ["c", "java", "python"]:
            stopped = False
            for n in SIZES:
                if stopped:
                    status.write(f"{lang},{n},SKIPPED,larger than a size over budget,\n")
                    status.flush()
                    continue
                print(f"{lang} n={n}", flush=True)
                result, detail, wall = run_configuration(lang, n, raw)
                status.write(f"{lang},{n},{result},{detail},{wall:.1f}\n")
                status.flush()
                stopped = result != "OK"


if __name__ == "__main__":
    main()
