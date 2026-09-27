# Can RECAP replace the invented scenarios? — a feasibility check

Jack Cushman (Harvard Library Innovation Lab) replied to a cold email on
31 Aug 2026 with two pointers, one of which was Free Law Project's **RECAP**
archive: real PACER dockets, which he suggested were "more realistic
litigation materials than the final court decisions in CAP".

The reply sent on 1 Sep said this would be checked "this week". This is that
check, run 2 Sep 2026 against the CourtListener REST API v4.

## Why this matters

Every scenario in this benchmark is synthetic. They were written by a Class 12
student with no lawful access to real matter files, which means the
environment measures performance against one person's imagination of
litigation. That is the single largest weakness in the project, and it is
stated as such in `SCENARIOS.md` and in the outreach that produced Cushman's
reply.

RECAP is the first lawful source of real litigation sequences found that could
substitute for invention.

## Method

CourtListener's search API, `type=r` (RECAP dockets), **no authentication
required** — these are public records. Counts are dockets and documents whose
text matches the query. Run 2 Sep 2026; counts drift as the archive grows.

```
https://www.courtlistener.com/api/rest/v4/search/?q=<query>&type=r
```

## Result: the doctrines this benchmark models are densely attested

| query | dockets | documents |
|---|---:|---:|
| `"privilege log"` | 45,193 | 132,859 |
| `"inadvertent" AND "privileged"` | 41,549 | 63,448 |
| `"work product" AND "motion to compel"` | 26,396 | 62,237 |
| `"in camera review" AND privilege` | 22,800 | 44,003 |
| `"motion to compel" AND "privilege log"` | 22,505 | 59,867 |
| `"spoliation"` | 22,613 | 76,769 |
| `"litigation hold"` | 8,698 | 16,681 |
| `"clawback" AND privilege` | 5,214 | 11,219 |
| `"crime-fraud exception"` | 3,007 | 8,854 |
| `"at issue waiver"` | 741 | 1,251 |

Mapped onto this repository's own scenario elements:

| LexCrisis element | real analogues (dockets) |
|---|---:|
| task_2 DOC-006, crime-fraud exception | 2,805 |
| task_2 DOC-007, at-issue waiver | 46,160 |
| task_2 inadvertent production / clawback | 26,368 |
| task_3 EVENT-001, litigation hold + spoliation | 2,328 |
| task_3 discovery response | 11,965 |
| task_1 conflict / disqualification | 5,089 |

Even the thinnest category has thousands of real matters. Scarcity is not the
constraint.

## The entries carry the structure the environment needs

The environment is sequential — it needs ordered events with dates, not just
topical text. Sampled dockets return exactly that. One example from
`State of Minnesota v. United States Department of Agriculture`
(D. Minn., 0:25-cv-04767):

```
#75  2026-06-01  Motion to Compel
#77  2026-06-01  Memorandum in Support of Motion
#80  2026-06-01  Proposed Order to Judge
```

Numbered entries, filing dates, and document types — the raw material for a
step sequence with real deadlines, rather than invented ones.

## Verdict

**Feasible, and worth doing.** RECAP can supply real procedural sequences to
replace invented ones, at a scale far beyond what this benchmark needs, from a
lawful public source.

## Three honest limits, before anyone gets excited

1. **RECAP is United States federal litigation. This benchmark is Indian.**
   The scenarios cite the Bharatiya Sakshya Adhiniyam and Bar Council of India
   rules. RECAP can seed *procedural shape* — the order and timing of a
   discovery fight — but it cannot supply Indian doctrine, and grafting US
   sequences onto Indian citations would deepen exactly the hybrid problem
   already flagged in `SCENARIOS.md` and put to Michał Araszkiewicz. Using
   RECAP well probably means being honest that the procedure modelled is US
   discovery, or doing the harder work of finding an Indian equivalent.
2. **These counts are keyword matches, not verified privilege disputes.**
   A docket mentioning "privilege log" is not necessarily a privilege dispute.
   Treat every number here as an upper bound; the usable fraction needs manual
   reading of a sample, which has not been done.
3. **Nothing has been built yet.** This establishes that the source is rich
   enough and lawfully reachable. It does not mean the substitution is done,
   and no claim should be made that this benchmark uses real data until it
   actually does.

## Reproducing this

```bash
curl -s "https://www.courtlistener.com/api/rest/v4/search/?q=%22privilege+log%22&type=r" \
  | python -c "import json,sys; d=json.load(sys.stdin); print(d['count'], d['document_count'])"
```

No API key needed for these queries.
