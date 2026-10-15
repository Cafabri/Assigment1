"""Builds the three implementations and runs the full benchmark under the declared budget.

Usage (from the repository root):  python run_experiment.py

Windows only: the C timer, the processor-affinity and peak-memory calls and the hardware
queries use Windows APIs. The toolchain paths below must point to the local installation.

Outputs:
  results/raw_measurements.csv  every timed run (warm-up, measured, check and memory runs)
  results/run_status.csv        OK / TIMEOUT / ERROR / SKIPPED for every process
  results/process_memory.csv    peak memory of every benchmark process (OS metric)
  results/allocation_times.csv  time of allocating the result matrix alone (Java, Python)
  results/environment.txt       hardware, OS, toolchain, JVM settings and protocol
"""
import ctypes
import ctypes.wintypes as wt
import os
import platform
import queue
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")

# Toolchain (adjust these paths on another machine)
GCC = r"C:\msys64\mingw64\bin\gcc.exe"
JDK_BIN = r"C:\Program Files\Microsoft\jdk-21.0.12.101-hotspot\bin"
JAVA = os.path.join(JDK_BIN, "java.exe")
JAVAC = os.path.join(JDK_BIN, "javac.exe")
PYTHON = sys.executable
CFLAGS = ["-O2", "-ffp-contract=off"]

# gcc needs its own bin directory on PATH to find the assembler and linker
os.environ["PATH"] = os.path.dirname(GCC) + os.pathsep + os.environ["PATH"]

# Declared before running: size sequence, repetitions and budget
SIZES = [10, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
REPETITIONS = 5                       # measured runs per process
PROCESSES = 3                         # independent processes for cheap configurations
CHEAP_SECONDS = 60                    # cheap = median run of the first process below this
WARMUPS = {"c": 2, "java": 10, "python": 1}   # Java: maximum, stops when stable (2 %)
PYTHON_LONG_N = 1500                  # Python from this size: 0 warm-ups, 3 measured runs
PYTHON_LONG_REPS = 3
MEMORY_MAX_N = 1000                   # largest size of the Python tracemalloc process
ALLOC_REPS = 5
RUN_BUDGET_SECONDS = 15 * 60          # maximum duration of a single run
MEMORY_BUDGET_GB = 30                 # largest size planned: 3 x 8 x 3000^2 B = 0.2 GB

LANGUAGES = ["c", "java", "python"]
COMMANDS = {
    "c": [os.path.join(ROOT, "c", "benchmark.exe")],
    "java": [JAVA, "-cp", os.path.join(ROOT, "java"), "Benchmark"],
    "python": [PYTHON, os.path.join(ROOT, "python", "benchmark.py")],
}

HEADER = ("language,n,dtype,threads,run_type,repetition,batch_size,time_seconds,"
          "memory_bytes,checksum,process")


# ---------------------------------------------------------------- processor affinity

def performance_cores():
    """Logical processors of the highest efficiency class (P-cores), or None."""
    k = ctypes.WinDLL("kernel32")
    size = ctypes.c_ulong(0)
    k.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buf = (ctypes.c_ubyte * size.value)()
    if not k.GetSystemCpuSetInformation(buf, size, ctypes.byref(size), None, 0):
        return None
    cpus, off = [], 0
    while off < size.value:                 # SYSTEM_CPU_SET_INFORMATION records
        length = int.from_bytes(bytes(buf[off:off + 4]), "little")
        cpus.append((buf[off + 14], buf[off + 18]))   # logical index, efficiency class
        off += length
    best = max(c for _, c in cpus)
    p_cores = sorted(lp for lp, c in cpus if c == best)
    return p_cores if len(p_cores) < len(cpus) else None


def pin_to_performance_cores():
    """Restricts this process (and therefore every child it starts) to the P-cores."""
    cpus = performance_cores()
    if not cpus:
        return "not set (no hybrid CPU detected)"
    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k.SetProcessAffinityMask(k.GetCurrentProcess(), sum(1 << c for c in cpus)):
        return "not set (SetProcessAffinityMask failed)"
    return f"P-cores only, logical processors {cpus[0]}-{cpus[-1]} ({len(cpus)})"


# ------------------------------------------------------------------- peak memory

class _Counters(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
        (f, ctypes.c_size_t) for f in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
            "PagefileUsage", "PeakPagefileUsage")]


_psapi = ctypes.WinDLL("psapi")
_psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_Counters), wt.DWORD]
PEAK_METRIC = "peak working set (GetProcessMemoryInfo)"


def wait_with_peak(proc):
    """Waits for the process and returns its peak working set in bytes, read through the
    process handle after the process has exited (the handle keeps the counters available)."""
    proc.wait()
    c = _Counters()
    c.cb = ctypes.sizeof(c)
    ok = _psapi.GetProcessMemoryInfo(int(proc._handle), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize if ok else None


# ------------------------------------------------------------------- environment

def run_text(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True)
    except OSError:
        return ""
    return (out.stdout.decode("oem", "replace") + out.stderr.decode("oem", "replace")).strip()


def first_line(cmd):
    text = run_text(cmd)
    return text.splitlines()[0] if text else "not found"


def hardware():
    ps = ("$p = Get-CimInstance Win32_Processor; $m = Get-CimInstance Win32_ComputerSystem;"
          "\"$($p.Name)|$($p.NumberOfCores)|$($p.NumberOfLogicalProcessors)|"
          "$($p.L2CacheSize)|$($p.L3CacheSize)|$($m.TotalPhysicalMemory)\"")
    name, cores, logical, l2, l3, ram = run_text(
        ["powershell", "-NoProfile", "-Command", ps]).split("|")
    p = performance_cores() or []
    return [f"cpu: {name.strip()}",
            f"cores: {cores} physical, {logical} logical",
            f"hybrid layout: {len(p)} logical processors on P-cores"
            + (f" ({p[0]}-{p[-1]}), {int(logical) - len(p)} on E-cores" if p else ""),
            f"cache: L2 {int(l2) // 1024} MB total, L3 {int(l3) // 1024} MB",
            f"ram: {int(ram) / 2**30:.1f} GiB"]


def power_plan():
    text = run_text(["powercfg", "/getactivescheme"])
    return "power plan: " + (text.split(":", 1)[-1].strip() if text else "unknown")


def jvm_settings():
    flags = ["UseG1GC", "UseParallelGC", "UseSerialGC", "MaxHeapSize", "InitialHeapSize",
             "ParallelGCThreads", "ConcGCThreads", "CICompilerCount", "TieredCompilation"]
    values = {}
    for line in run_text([JAVA, "-XX:+PrintFlagsFinal", "-version"]).splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1] in flags:
            values[parts[1]] = parts[3]
    out = [f"jvm.{k}: {values.get(k, 'n/a')}" for k in flags]
    for k in ("MaxHeapSize", "InitialHeapSize"):
        if values.get(k, "").isdigit():
            out[flags.index(k)] += f" ({int(values[k]) / 2**20:.0f} MiB)"
    return out


def library_versions():
    versions = []
    for mod in ("numpy", "pandas", "matplotlib"):
        try:
            versions.append(f"{mod} {__import__(mod).__version__}")
        except ImportError:
            versions.append(f"{mod} not installed")
    return ", ".join(versions)


def save_environment(affinity):
    java_version = run_text([JAVA, "-version"]).splitlines()
    lines = [f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"os: {platform.platform()}"] + hardware() + [
        f"affinity of the runner and all benchmark processes: {affinity}",
        power_plan(),
        f"c compiler: {first_line([GCC, '--version'])}  (flags: {' '.join(CFLAGS)})",
        f"java: {' / '.join(java_version)}",
        "jvm options: none (default JIT and garbage collector)",
        *jvm_settings(),
        f"python: {platform.python_implementation()} {platform.python_version()} "
        f"(benchmark: standard library only)",
        f"analysis libraries: {library_versions()}",
        "kernel threads: 1 in every language",
        f"peak memory metric: {PEAK_METRIC}",
        f"sizes: {SIZES}",
        f"measured runs per process: {REPETITIONS} (Python n >= {PYTHON_LONG_N}: "
        f"{PYTHON_LONG_REPS})",
        f"processes: {PROCESSES} interleaved when the first process has a median below "
        f"{CHEAP_SECONDS} s, otherwise 1",
        f"warm-ups: C {WARMUPS['c']}, Java until two consecutive runs differ by < 2 % "
        f"(min 2, max {WARMUPS['java']}), Python {WARMUPS['python']} (0 for n >= "
        f"{PYTHON_LONG_N})",
        "batching: n <= 100 timed in batches lasting >= 10 ms, time per multiplication "
        "reported, checksum from a separate single call",
        f"budget: {RUN_BUDGET_SECONDS} s per run, {MEMORY_BUDGET_GB} GB memory",
    ]
    with open(os.path.join(RESULTS, "environment.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


# ------------------------------------------------------------------- running

def build():
    subprocess.run([GCC, *CFLAGS, "-o", "benchmark.exe", "matmul.c", "benchmark.c"],
                   cwd=os.path.join(ROOT, "c"), check=True)
    subprocess.run([JAVAC, "MatMul.java", "Benchmark.java"],
                   cwd=os.path.join(ROOT, "java"), check=True)


def run_process(cmd, prefix):
    """Runs one benchmark process, killing it when a single run exceeds the budget.

    Returns (status, detail, wall seconds, output lines, peak memory in bytes)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            cwd=ROOT)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l) for l in proc.stdout] + [lines.put(None)],
                     daemon=True).start()

    out = []
    start = last = time.monotonic()
    while True:
        try:
            line = lines.get(timeout=max(0.0, RUN_BUDGET_SECONDS - (time.monotonic() - last)))
        except queue.Empty:
            proc.kill()
            proc.wait()
            return ("TIMEOUT", f"a run exceeded {RUN_BUDGET_SECONDS} s",
                    time.monotonic() - start, out, None)
        if line is None:
            break
        if line.startswith(prefix):
            out.append(line.strip())
            last = time.monotonic()
            print("   ", line.strip(), flush=True)

    peak = wait_with_peak(proc)
    wall = time.monotonic() - start
    if proc.returncode != 0:
        detail = proc.stderr.read().strip().replace("\n", " ").replace(",", ";")
        return "ERROR", f"exit code {proc.returncode}: {detail}", wall, out, peak
    return "OK", "", wall, out, peak


class Runner:
    def __init__(self):
        self.raw = open(os.path.join(RESULTS, "raw_measurements.csv"), "w", newline="")
        self.status = open(os.path.join(RESULTS, "run_status.csv"), "w", newline="")
        self.memory = open(os.path.join(RESULTS, "process_memory.csv"), "w", newline="")
        self.raw.write(HEADER + "\n")
        self.status.write("language,n,process,mode,status,detail,wall_seconds\n")
        self.memory.write("language,n,process,mode,peak_bytes,metric\n")

    def benchmark(self, lang, n, process, mode="run"):
        """Runs one process; returns its status and the median of its measured runs."""
        if mode == "memory":
            args = ["memory", str(n)]
        else:
            long_python = lang == "python" and n >= PYTHON_LONG_N
            warmups = 0 if long_python else WARMUPS[lang]
            reps = PYTHON_LONG_REPS if long_python else REPETITIONS
            args = ["run", str(n), str(warmups), str(reps)]
        print(f"{lang} n={n} process={process} {mode}", flush=True)
        status, detail, wall, out, peak = run_process(COMMANDS[lang] + args, lang + ",")

        for line in out:
            self.raw.write(f"{line},{process}\n")
        self.status.write(f"{lang},{n},{process},{mode},{status},{detail},{wall:.1f}\n")
        if peak is not None and status == "OK":
            self.memory.write(f"{lang},{n},{process},{mode},{peak},{PEAK_METRIC}\n")
        for f in (self.raw, self.status, self.memory):
            f.flush()

        times = [float(l.split(",")[7]) for l in out if l.split(",")[4] == "measured"]
        return status, statistics.median(times) if times else None

    def skip(self, lang, n):
        self.status.write(f"{lang},{n},1,run,SKIPPED,larger than a size over budget,\n")
        self.status.flush()


def allocation_times(sizes):
    with open(os.path.join(RESULTS, "allocation_times.csv"), "w", newline="") as f:
        f.write("language,n,repetition,time_seconds\n")
        for lang in ["java", "python"]:
            for n in sizes[lang]:
                print(f"{lang} n={n} alloc", flush=True)
                status, _, _, out, _ = run_process(
                    COMMANDS[lang] + ["alloc", str(n), str(ALLOC_REPS)], lang + ",")
                for line in out:
                    f.write(line + "\n")
                f.flush()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    affinity = pin_to_performance_cores()
    build()
    save_environment(affinity)

    runner = Runner()
    cheap, completed = [], {lang: [] for lang in LANGUAGES}

    # First process of every configuration, language by language, with the budget stop rule
    for lang in LANGUAGES:
        stopped = False
        for n in SIZES:
            if stopped:
                runner.skip(lang, n)
                continue
            status, median = runner.benchmark(lang, n, 1)
            stopped = status != "OK"
            if stopped:
                continue
            completed[lang].append(n)
            if median is not None and median < CHEAP_SECONDS:
                cheap.append((lang, n))
            if lang == "python" and n <= MEMORY_MAX_N:
                runner.benchmark(lang, n, 1, mode="memory")

    # Further independent processes of the cheap configurations, interleaved
    for process in range(2, PROCESSES + 1):
        for n in SIZES:
            for lang in LANGUAGES:
                if (lang, n) in cheap:
                    runner.benchmark(lang, n, process)

    allocation_times(completed)


if __name__ == "__main__":
    main()
