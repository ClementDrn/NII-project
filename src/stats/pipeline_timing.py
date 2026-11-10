"""Deterministic generation/simulation pipeline timing; standard library only.

Each independent scenario/repetition follows G0 -> S0 -> G1 -> S1 -> G2 -> S2.
Different chains may overlap. Generation and simulation have separate worker
pools and FIFO queues. No iteration-wide barrier is imposed.

Defaults use measured totals for 25 sequential generation requests per stage
and 50 simulations. All jobs at a given stage take the corresponding mean:
these are planning estimates, not measured pipelined runtimes or uncertainty
bounds. Increasing concurrency assumes unchanged per-request service times.

Optional first-batch delivery replaces G0 with external response arrivals. A
zero delivery spread assumes ALL first instructions arrive at the stated time;
the first-response time alone does not establish that assumption. Subsequent
generation uses the normal worker pool, with no shared batch/API contention.

The CLI defaults to three consecutive 25-scenario repetitions (225 simulations).
Use --joint-repetitions to allow all repetition chains to overlap as well.
"""

import argparse
import csv
from collections import deque
import heapq
from itertools import count
import json
import math
from pathlib import Path


def seconds(text):
    """Parse seconds or HH:MM:SS, including fractional seconds."""
    parts = str(text).split(":")
    if len(parts) == 1:
        value = float(parts[0])
    elif len(parts) == 3:
        h, m, s = map(float, parts)
        if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
            raise ValueError("Use nonnegative HH:MM:SS with minutes/seconds below 60")
        value = h * 3600 + m * 60 + s
    else:
        raise ValueError("Use seconds or HH:MM:SS")
    if not math.isfinite(value) or value < 0:
        raise ValueError("Duration must be finite and nonnegative")
    return value


def format_time(value):
    h, remainder = divmod(value, 3600)
    m, s = divmod(remainder, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def simulate_pipeline(
    scenarios=25,
    repetitions=1,
    generation_seconds=(276.28, 839.72, 999.88),
    simulation_seconds=585.94,
    generation_workers=1,
    simulation_workers=1,
    first_batch_seconds=None,
    first_batch_spread=0.0,
):
    """Return (summary, task log); all durations and log timestamps are seconds.

    Each repetition is a separate independent chain; --repetitions 3 schedules
    all 75 chains together, rather than running three campaigns consecutively.
    First-batch timing applies to all scenario/repetition chains supplied.
    """
    for value in [scenarios, repetitions, generation_workers, simulation_workers]:
        if not isinstance(value, int) or value < 1:
            raise ValueError("Counts must be positive integers")
    durations = tuple(float(v) for v in generation_seconds)
    if not durations or any(not math.isfinite(v) or v <= 0 for v in durations):
        raise ValueError("Generation durations must be finite and positive")
    if not math.isfinite(simulation_seconds) or simulation_seconds <= 0:
        raise ValueError("Simulation duration must be finite and positive")
    if not math.isfinite(first_batch_spread) or first_batch_spread < 0:
        raise ValueError("Batch response spread must be finite and nonnegative")
    if first_batch_seconds is not None and (
        not math.isfinite(first_batch_seconds) or first_batch_seconds < 0
    ):
        raise ValueError("Batch response time must be finite and nonnegative")
    if first_batch_seconds is None and first_batch_spread:
        raise ValueError("Batch spread requires a first-batch response time")

    chains = [
        (scenario, repetition)
        for repetition in range(repetitions)
        for scenario in range(scenarios)
    ]
    queues = {"generation": deque(), "simulation": deque()}
    free = {
        "generation": list(range(generation_workers)),
        "simulation": list(range(simulation_workers)),
    }
    events, log = [], []
    sequence = count()
    if first_batch_seconds is None:
        for chain in chains:
            queues["generation"].append((chain, 0, 0.0))
    else:
        for index, chain in enumerate(chains):
            arrival = first_batch_seconds + first_batch_spread * index / max(
                1, len(chains) - 1
            )
            heapq.heappush(
                events, (arrival, next(sequence), "batch_response", chain, 0, -1)
            )

    def dispatch(now):
        for kind in ["generation", "simulation"]:
            while free[kind] and queues[kind]:
                chain, iteration, ready = queues[kind].popleft()
                worker = heapq.heappop(free[kind])
                duration = (
                    durations[iteration] if kind == "generation" else simulation_seconds
                )
                end = now + duration
                log.append(
                    {
                        "scenario": chain[0],
                        "repetition": chain[1],
                        "iteration": iteration,
                        "kind": kind,
                        "worker": worker,
                        "ready_seconds": ready,
                        "start_seconds": now,
                        "end_seconds": end,
                        "wait_seconds": now - ready,
                    }
                )
                heapq.heappush(
                    events, (end, next(sequence), kind, chain, iteration, worker)
                )

    now = 0.0
    dispatch(now)
    while events:
        now = events[0][0]
        # Complete simultaneous events before assigning any newly free worker.
        while events and events[0][0] == now:
            _, _, kind, chain, iteration, worker = heapq.heappop(events)
            if kind == "batch_response":
                log.append(
                    {
                        "scenario": chain[0],
                        "repetition": chain[1],
                        "iteration": 0,
                        "kind": kind,
                        "worker": -1,
                        "ready_seconds": now,
                        "start_seconds": now,
                        "end_seconds": now,
                        "wait_seconds": 0.0,
                    }
                )
                queues["simulation"].append((chain, 0, now))
            else:
                heapq.heappush(free[kind], worker)
                if kind == "generation":
                    queues["simulation"].append((chain, iteration, now))
                elif iteration + 1 < len(durations):
                    queues["generation"].append((chain, iteration + 1, now))
        dispatch(now)

    expected = len(chains) * len(durations)
    if sum(row["kind"] == "simulation" for row in log) != expected:
        raise RuntimeError("Some simulations were not scheduled")
    summary = {
        "scenarios": scenarios,
        "repetitions": repetitions,
        "simulations": expected,
        "generation_workers": generation_workers,
        "simulation_workers": simulation_workers,
        "first_batch_seconds": first_batch_seconds,
        "first_batch_spread_seconds": first_batch_spread,
        "makespan_seconds": now,
        "makespan_hours": now / 3600,
        "makespan_hms": format_time(now),
    }
    for kind, workers in [
        ("generation", generation_workers),
        ("simulation", simulation_workers),
    ]:
        busy = sum(
            row["end_seconds"] - row["start_seconds"]
            for row in log
            if row["kind"] == kind
        )
        summary[f"{kind}_worker_hours"] = busy / 3600
        summary[f"{kind}_utilization"] = busy / (workers * now)
    return summary, sorted(
        log, key=lambda row: (row["start_seconds"], row["kind"], row["scenario"])
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenarios", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--joint-repetitions",
        action="store_true",
        help="Overlap repetition chains instead of consecutive campaigns",
    )
    parser.add_argument(
        "--generation-totals",
        nargs="+",
        type=seconds,
        default=[6907.0, 20993.0, 24997.0],
        metavar="HH:MM:SS",
    )
    parser.add_argument("--generation-reference-count", type=int, default=25)
    parser.add_argument("--simulation-total", type=seconds, default=29297.0)
    parser.add_argument("--simulation-reference-count", type=int, default=50)
    parser.add_argument("--generation-workers", type=int, default=1)
    parser.add_argument("--simulation-workers", type=int, default=1)
    parser.add_argument(
        "--first-batch-seconds",
        type=seconds,
        default=None,
        help="First G0 response time; zero spread assumes all G0 responses arrive together",
    )
    parser.add_argument(
        "--first-batch-spread",
        type=seconds,
        default=0.0,
        help="Assumed interval from first to last G0 response; arrivals spaced evenly",
    )
    parser.add_argument(
        "--output", type=Path, help="Task-log CSV path; summary saved alongside as JSON"
    )
    args = parser.parse_args()
    if min(args.generation_reference_count, args.simulation_reference_count) < 1:
        parser.error("Reference counts must be positive")
    generation = [v / args.generation_reference_count for v in args.generation_totals]
    sim = args.simulation_total / args.simulation_reference_count
    try:
        summary, log = simulate_pipeline(
            args.scenarios,
            args.repetitions if args.joint_repetitions else 1,
            generation,
            sim,
            args.generation_workers,
            args.simulation_workers,
            args.first_batch_seconds,
            args.first_batch_spread,
        )
        if args.repetitions < 1:
            raise ValueError("Repetitions must be positive")
        if not args.joint_repetitions:
            period = summary["makespan_seconds"]
            log = [
                dict(
                    row,
                    repetition=rep,
                    ready_seconds=row["ready_seconds"] + rep * period,
                    start_seconds=row["start_seconds"] + rep * period,
                    end_seconds=row["end_seconds"] + rep * period,
                )
                for rep in range(args.repetitions)
                for row in log
            ]
            summary["repetitions"] = args.repetitions
            summary["simulations"] *= args.repetitions
            for key in [
                "makespan_seconds",
                "makespan_hours",
                "generation_worker_hours",
                "simulation_worker_hours",
            ]:
                summary[key] *= args.repetitions
            summary["makespan_hms"] = format_time(summary["makespan_seconds"])
        summary["repetition_scheduling"] = (
            "joint" if args.joint_repetitions else "consecutive"
        )
    except ValueError as error:
        parser.error(str(error))
    sequential = (
        args.scenarios * args.repetitions * (sum(generation) + len(generation) * sim)
    )
    summary["unbatched_sequential_reference_hours"] = sequential / 3600
    summary["generation_service_seconds_per_stage"] = generation
    summary["simulation_service_seconds"] = sim
    print("Deterministic timing estimate (not measured runtime)")
    print(
        f"{summary['simulations']} simulations; generation workers={args.generation_workers}; simulation workers={args.simulation_workers}"
    )
    print(
        "Generation minutes/request by iteration:",
        ", ".join(f"{x/60:.3f}" for x in generation),
    )
    print(f"Simulation minutes/run: {sim/60:.3f}")
    print(
        f"Unbatched sequential reference: {format_time(sequential)} ({sequential/3600:.6f} h)"
    )
    print(
        f"Pipeline makespan: {summary['makespan_hms']} ({summary['makespan_hours']:.6f} h)"
    )
    print(f"Simulator utilization: {summary['simulation_utilization']:.1%}")
    if args.first_batch_seconds is not None:
        print(
            "G0 batch delivery is assumed, not measured; later generations use the worker pool."
        )
    if args.output:
        if args.output.suffix.lower() != ".csv":
            parser.error("--output must end in .csv")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(log[0]))
            writer.writeheader()
            writer.writerows(log)
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(f"Saved {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
