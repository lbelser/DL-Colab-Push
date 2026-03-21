"""
Experiment & decision tracking for the WikiArt project.

Every modelling choice — what we tried, why, and what happened — is
logged here so we can reconstruct the full narrative for the report.

Two complementary logs:
  - **Experiments**: hyperparameters + metrics for every training run
  - **Decisions**: the reasoning behind each design choice, linked to
    the experiments that motivated them

Both are persisted to JSON so nothing is lost between notebook sessions.
"""

import json
import os
from datetime import datetime

from config import OUTPUT_DIR

EXPERIMENT_LOG_PATH = os.path.join(OUTPUT_DIR, "experiment_log.json")
DECISION_LOG_PATH = os.path.join(OUTPUT_DIR, "decision_log.json")


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────
def _load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def _save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ──────────────────────────────────────────────
# 1. Experiment logging
# ──────────────────────────────────────────────
def log_experiment(name, config, results, notes=""):
    """
    Record a training experiment with its configuration and results.

    Parameters
    ----------
    name    : str — identifier (e.g. "ResNet50_FineTuned_v2")
    config  : dict — hyperparameters (lr, epochs, augmentation, etc.)
    results : dict — metrics (accuracy, top-3, top-5, etc.)
    notes   : str — free-text observations about this run

    Example
    -------
    log_experiment(
        name="Baseline_CNN",
        config={"epochs": 30, "lr": 1e-3, "augmentation": "basic"},
        results={"test_accuracy": 0.482, "top3": 0.715, "top5": 0.815},
        notes="Underfitting — model too shallow for 23-class problem",
    )
    """
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
    """Return all logged experiments as a list of dicts."""
    return _load_json(EXPERIMENT_LOG_PATH)


# ──────────────────────────────────────────────
# 2. Decision logging
# ──────────────────────────────────────────────
def log_decision(title, context, reasoning, action, outcome=None):
    """
    Record a design decision with its full rationale.

    Parameters
    ----------
    title     : str — short label (e.g. "Switch from EfficientNetB0 to B2")
    context   : str — what situation prompted this decision
    reasoning : str — why we chose this path over alternatives
    action    : str — what we concretely did
    outcome   : str or None — result after implementing (fill in later)

    Example
    -------
    log_decision(
        title="Switch from EfficientNetB0 to B2",
        context="B0 achieved 73.9% vs ResNet50's 78.8% despite fine-tuning",
        reasoning="B0 has only 5.3M params vs ResNet50's 25.6M — likely "
                  "too small to capture the diversity of 23 artistic styles",
        action="Upgrade to EfficientNetB2 (9.2M params, 260×260 native res)",
    )
    """
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
    """
    Fill in the outcome of a previously logged decision
    once the experiment results are available.
    """
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
    """Return all logged decisions as a list of dicts."""
    return _load_json(DECISION_LOG_PATH)


# ──────────────────────────────────────────────
# 3. Pretty-print for the notebook
# ──────────────────────────────────────────────
def print_experiment_summary():
    """Print a formatted table of all experiments."""
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
    """Print the full decision log in a report-friendly format."""
    decisions = get_decisions()
    if not decisions:
        print("No decisions logged yet.")
        return

    print(f"\n{'='*80}")
    print(f"{'DECISION LOG':^80}")
    print(f"{'='*80}")
    for d in decisions:
        print(f"\n{'─'*80}")
        print(f"  Decision #{d['id']}: {d['title']}")
        print(f"{'─'*80}")
        print(f"  Context:   {d['context']}")
        print(f"  Reasoning: {d['reasoning']}")
        print(f"  Action:    {d['action']}")
        if d.get("outcome"):
            print(f"  Outcome:   {d['outcome']}")
        else:
            print(f"  Outcome:   (pending)")
