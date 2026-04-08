"""
Experiment & decision tracking — logs everything to JSON
so we can reconstruct the narrative for the report.
"""

import json
import os
from datetime import datetime

from config import OUTPUT_DIR

EXPERIMENT_LOG_PATH = os.path.join(OUTPUT_DIR, "experiment_log.json")
DECISION_LOG_PATH = os.path.join(OUTPUT_DIR, "decision_log.json")


def _load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ──────────────────────────────────────────────
# Experiment logging
# ──────────────────────────────────────────────
def log_experiment(name, config, results, notes=""):
    """Log a training run: config + results + notes."""
    experiments = _load_json(EXPERIMENT_LOG_PATH)
    entry = {
        "id": len(experiments) + 1,
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": results,
        "notes": notes,
    }
    experiments.append(entry)
    _save_json(EXPERIMENT_LOG_PATH, experiments)
    print(f"[Tracker] Logged experiment #{entry['id']}: {name}")
    return entry


def get_experiments():
    return _load_json(EXPERIMENT_LOG_PATH)


# ──────────────────────────────────────────────
# Decision logging
# ──────────────────────────────────────────────
def log_decision(title, context, reasoning, action, outcome=None):
    """Log a design decision with context and reasoning."""
    decisions = _load_json(DECISION_LOG_PATH)
    entry = {
        "id": len(decisions) + 1,
        "title": title,
        "timestamp": datetime.now().isoformat(),
        "context": context,
        "reasoning": reasoning,
        "action": action,
        "outcome": outcome,
    }
    decisions.append(entry)
    _save_json(DECISION_LOG_PATH, decisions)
    print(f"[Decision] #{entry['id']}: {title}")
    return entry


def update_decision_outcome(decision_id, outcome):
    """Fill in the outcome of a previously logged decision."""
    decisions = _load_json(DECISION_LOG_PATH)
    for d in decisions:
        if d["id"] == decision_id:
            d["outcome"] = outcome
            _save_json(DECISION_LOG_PATH, decisions)
            print(f"[Decision] Updated outcome for #{decision_id}: {d['title']}")
            return d
    print(f"[Decision] Warning: decision #{decision_id} not found")
    return None


def get_decisions():
    return _load_json(DECISION_LOG_PATH)


# ──────────────────────────────────────────────
# Pretty-print for the notebook
# ──────────────────────────────────────────────
def print_experiment_summary():
    experiments = get_experiments()
    if not experiments:
        print("No experiments logged yet.")
        return

    print(f"\n{'='*80}")
    print(f"{'EXPERIMENT LOG':^80}")
    print(f"{'='*80}")
    for exp in experiments:
        acc = exp["results"].get("test_accuracy", exp["results"].get("val_accuracy", "N/A"))
        acc_str = f"{acc:.1%}" if isinstance(acc, float) else acc
        print(f"\n  #{exp['id']} {exp['name']}")
        print(f"     Config:  {exp['config']}")
        print(f"     Result:  {acc_str}")
        if exp.get("notes"):
            print(f"     Notes:   {exp['notes']}")


def print_decision_log():
    decisions = get_decisions()
    if not decisions:
        print("No decisions logged yet.")
        return

    print(f"\n{'='*80}")
    print(f"{'DECISION LOG':^80}")
    print(f"{'='*80}")
    for d in decisions:
        print(f"\n{'-'*80}")
        print(f"  Decision #{d['id']}: {d['title']}")
        print(f"{'-'*80}")
        print(f"  Context:   {d['context']}")
        print(f"  Reasoning: {d['reasoning']}")
        print(f"  Action:    {d['action']}")
        if d.get("outcome"):
            print(f"  Outcome:   {d['outcome']}")
        else:
            print(f"  Outcome:   (pending)")
