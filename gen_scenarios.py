#!/usr/bin/env python3
"""Render the three LexCrisis scenarios as readable prose.

The benchmark's scenarios live as Python data in lexcrisis_env/tasks.py, which
is the wrong format for a lawyer to review. This writes SCENARIOS.md from those
structures directly, so the document cannot drift away from what the graders
actually score.

    python gen_scenarios.py > SCENARIOS.md
"""

from lexcrisis_env import tasks as T

OUT = []
w = OUT.append


def rule(client_a, client_b):
    return T.CONFLICT_RULES.get(frozenset({client_a, client_b}))


w("# LexCrisis scenarios")
w("")
w("The full contents of the three environments, rendered from")
w("`lexcrisis_env/tasks.py` by `gen_scenarios.py`, so this document cannot")
w("drift from the data the graders actually score.")
w("")
w("Every scenario is **synthetic**. They were written by a Class 12 student in")
w("India with no access to real matter files and no experience of practice.")
w("Their internal legal coherence is bounded by that.")
w("")
w("**Known defect, stated up front:** the privilege scenario cites the Indian")
w("Evidence Act 1872, which was repealed by the Bharatiya Sakshya Adhiniyam")
w("2023 with effect from 1 July 2024. IEA s.126 is now BSA s.132, IEA s.129 is")
w("BSA s.134, and IEA s.45 is BSA s.39. Every statutory reference below is")
w("therefore to law that was not in force when the benchmark was built. See")
w("issue #19.")
w("")

# ---------------------------------------------------------------- task 1
d = T.TASK_DEFINITIONS["task_1"]
w(f"## {d.name} (`{d.task_id}`)")
w("")
w(f"{d.description}")
w("")
w(f"Horizon: {d.max_steps} steps. Difficulty: {d.difficulty}.")
w("")
w("### The prospective clients")
w("")
for c in T.CLIENTS:
    decision = T.CONFLICT_DECISIONS.get(c.client_id, "?")
    w(f"**{c.client_id} - {c.name}** ({c.client_type}) - *{decision}*")
    w("")
    w(f"{c.summary}")
    w("")
    w(f"{c.details}")
    w("")
    if c.relationships:
        w(f"Relationships: {'; '.join(c.relationships)}")
        w("")

w("### The conflicts the grader expects to be found")
w("")
w("| Clients | Rule |")
w("|---|---|")
for pair, rule_name in T.CONFLICT_RULES.items():
    a, b = sorted(pair)
    w(f"| {a} + {b} | {rule_name} |")
w("")
w("`decline` is correct for CLIENT-003, CLIENT-004 and CLIENT-005; the other")
w("three are `accept`. The agent is scored on reaching those decisions, on")
w("reviewing before deciding, and on producing the conflict record.")
w("")

# ---------------------------------------------------------------- task 2
d = T.TASK_DEFINITIONS["task_2"]
w(f"## {d.name} (`{d.task_id}`)")
w("")
w(f"{d.description}")
w("")
w(f"Horizon: {d.max_steps} steps. Difficulty: {d.difficulty}.")
w("")
w("Ten documents. For each the agent must classify the privilege, name the")
w("doctrine, decide produce or withhold, and identify any exception. Producing")
w("a privileged document carries a -0.12 penalty.")
w("")
for doc in T.PRIVILEGE_DOCUMENTS:
    gt = T.PRIVILEGE_GROUND_TRUTH.get(doc.doc_id, {})
    w(f"### {doc.doc_id} - {doc.title}")
    w("")
    w("```")
    for line in doc.content.splitlines():
        w(line)
    w("```")
    w("")
    w(f"- Classification: `{gt.get('classification', '?')}`")
    doctrine = gt.get("doctrine") or "(none)"
    w(f"- Doctrine: `{doctrine}`")
    w(f"- Action: `{gt.get('action', '?')}`")
    w(f"- Exception: `{gt.get('exception', '?')}`")
    if doc.doc_id in T.WAIVER_EVENTS:
        w(f"- Waiver event: `{T.WAIVER_EVENTS[doc.doc_id]}`")
    w("")

w("Three of the ten carry no doctrine at all. That is the hole that made an")
w("empty submission score 0.0609 - the grader read an absent field as a correct")
w("answer. Filed as issue #1, fixed in PR #9.")
w("")

# ---------------------------------------------------------------- task 3
d = T.TASK_DEFINITIONS["task_3"]
w(f"## {d.name} (`{d.task_id}`)")
w("")
w(f"{d.description}")
w("")
w(f"Horizon: {d.max_steps} steps. Difficulty: {d.difficulty}.")
w("")
gt = T.CRISIS_GROUND_TRUTH
for ev in T.CRISIS_EVENTS:
    w(f"### {ev.event_id} - {ev.title}")
    w("")
    w(f"Type: `{ev.event_type}`. Deadline: step {ev.deadline_step}.")
    w("")
    w(f"{ev.content}")
    w("")
    w(f"Consequence if missed: {ev.consequence}")
    w("")
    req = gt["deadlines"].get(ev.event_id, {}).get("required_action")
    if req:
        w(f"Required action: `{req}`")
    if ev.event_id in gt["ethical_issues"]:
        w("Flagged as an **ethical** issue rather than a deadline.")
    if ev.event_id in gt["adversarial_items"]:
        w("Flagged as **adversarial** - the agent is being pushed toward a wrong act.")
    w("")

w("Expected priority order:")
w("")
w(f"{' -> '.join(gt['priority_order'])}")
w("")

print("\n".join(OUT))
