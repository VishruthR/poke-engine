import sys
import os
import glob
import math
import statistics


METRICS = ["elapsed_ms", "iterations", "nodes_expanded", "max_depth", "avg_depth"]
DERIVED = ["iters_per_ms", "nodes_per_ms", "nodes_per_iter"]

COMPARE_DURATIONS = [100, 500]


def parse_line(line):
    out = {}
    for tok in line.strip().split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            pass
    return out


def key_of(path):
    """Return a group key for a log file, e.g. 'ucb_100', 'ucb_500', 'exp3'."""
    name = os.path.basename(path)
    stem = name[:-4] if name.endswith(".log") else name
    parts = stem.split("_")
    if "ucb" in parts:
        # New format: poke_mcts_stats_ucb_{pid}_{duration} — duration is last part
        if parts[-1].isdigit():
            return f"ucb_{parts[-1]}"
        return "ucb"
    if "exp3" in parts:
        return "exp3"
    return None


def summarize(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
    }


def fmt(x):
    if isinstance(x, int):
        return f"{x:>14d}"
    return f"{x:>14.4f}"


def print_table(metric, summaries):
    print(f"\n=== {metric} ===")
    cols = ["n", "mean", "stdev", "min", "median", "max"]
    header = f"{'method':<12}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for method, s in sorted(summaries.items()):
        if s["n"] == 0:
            print(f"{method:<12}  (no data)")
            continue
        row = f"{method:<12}" + "".join(fmt(s[c]) for c in cols)
        print(row)


def welch_t(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return None
    return (ma - mb) / se


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_stats.py <logs_dir>")
        sys.exit(1)
    logs_dir = sys.argv[1]

    all_keys = set()
    data = {}
    file_counts = {}
    line_counts = {}

    for path in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
        key = key_of(path)
        if key is None:
            continue
        if key not in data:
            all_keys.add(key)
            data[key] = {m: [] for m in METRICS + DERIVED}
            file_counts[key] = 0
            line_counts[key] = 0
        file_counts[key] += 1
        with open(path) as f:
            for line in f:
                rec = parse_line(line)
                if not all(k in rec for k in METRICS):
                    continue
                line_counts[key] += 1
                for m in METRICS:
                    data[key][m].append(rec[m])
                ms = rec["elapsed_ms"]
                it = rec["iterations"]
                nodes = rec["nodes_expanded"]
                if ms > 0:
                    data[key]["iters_per_ms"].append(it / ms)
                    data[key]["nodes_per_ms"].append(nodes / ms)
                if it > 0:
                    data[key]["nodes_per_iter"].append(nodes / it)

    for key in sorted(all_keys):
        print(f"  {key}: {file_counts[key]} files, {line_counts[key]} lines")

    summaries_by_metric = {}
    for m in METRICS + DERIVED:
        summaries_by_metric[m] = {k: summarize(data[k][m]) for k in sorted(all_keys)}
        print_table(m, summaries_by_metric[m])

    # Compare ucb_100 vs ucb_500
    key_a = f"ucb_{COMPARE_DURATIONS[0]}"
    key_b = f"ucb_{COMPARE_DURATIONS[1]}"
    if key_a in all_keys and key_b in all_keys:
        print(f"\n=== {key_b} vs {key_a} comparison (mean ratios and Welch t-stat) ===")
        print(f"{'metric':<18}{key_a + '_mean':>16}{key_b + '_mean':>16}{'ratio(b/a)':>12}{'welch_t':>12}")
        print("-" * 74)
        for m in METRICS + DERIVED:
            s_a = summaries_by_metric[m][key_a]
            s_b = summaries_by_metric[m][key_b]
            if s_a["n"] == 0 or s_b["n"] == 0:
                continue
            ratio = s_b["mean"] / s_a["mean"] if s_a["mean"] != 0 else float("nan")
            t = welch_t(data[key_b][m], data[key_a][m])
            t_str = f"{t:>12.3f}" if t is not None else f"{'n/a':>12}"
            print(f"{m:<18}{s_a['mean']:>16.4f}{s_b['mean']:>16.4f}{ratio:>12.4f}{t_str}")


if __name__ == "__main__":
    main()
