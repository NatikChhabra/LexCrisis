# LexCrisis scenarios

The full contents of the three environments, rendered from
`lexcrisis_env/tasks.py` by `gen_scenarios.py`, so this document cannot
drift from the data the graders actually score.

Every scenario is **synthetic**. They were written by a Class 12 student in
India with no access to real matter files and no experience of practice.
Their internal legal coherence is bounded by that.

**Statute:** citations are to the **Bharatiya Sakshya Adhiniyam 2023**, in
force since 1 July 2024. Until issue #19 was fixed they were to the repealed
Indian Evidence Act 1872, and the grader marked an agent wrong for citing the
law actually in force. The repealed forms - IEA s.126, s.129, s.45 - still
earn full credit, so a model trained on pre-2024 material is not punished.

**The open question**, which renumbering does not answer: the classification
vocabulary includes `work_product`, a United States doctrine with no clean
Indian equivalent, and the procedure throughout - privilege logs,
produce/withhold calls - is US-style discovery. This may be US procedure
wearing Indian citations.

## Conflict-Safe Client Intake (`task_1`)

A law firm intake team must decide which prospective clients to accept in a fast-moving product-liability crisis. The current scenario uses a pharmaceutical case study, but the workflow applies more broadly to regulated-industry litigation. The agent needs to identify conflicts of interest, cite the right Bar Council of India rule, and make intake decisions that a real conflicts team would stand behind.

Horizon: 24 steps. Difficulty: easy.

### The prospective clients

**CLIENT-001 - Ravi Sharma** (plaintiff) - *accept*

Severe liver injury claimant tied to the Veridex litigation wave.

Ravi took Veridex for six months and now alleges acute hepatic failure. He wants Sterling & Associates to pursue NovaChem India and related entities.

Relationships: Adverse to NovaChem India; Purchased through MedDistro Ltd

**CLIENT-002 - Priya Patel** (plaintiff) - *accept*

Potential class representative coordinating dozens of claimants.

Priya is organizing 40+ patients with similar injuries and wants coordinated representation against NovaChem India.

Relationships: Adverse to NovaChem India

**CLIENT-003 - NovaChem India Pvt Ltd** (defendant) - *decline*

Drug manufacturer seeking defense counsel.

NovaChem manufactured Veridex and faces parallel product liability matters, consumer claims, and reputational exposure.

Relationships: Adverse to patient plaintiffs; Potential indemnity dispute with MedDistro Ltd

**CLIENT-004 - Dr. Anil Kapoor** (prescriber) - *decline*

Physician co-defendant in plaintiff suits.

Dr. Kapoor prescribed Veridex to Ravi Sharma and is being sued alongside NovaChem for malpractice and failure-to-warn theories.

Relationships: Aligned with NovaChem in some filings; Adverse to Ravi Sharma

**CLIENT-005 - MedDistro Ltd** (distributor) - *decline*

Distributor seeking indemnity from NovaChem.

MedDistro distributed the affected batch and is exploring indemnification claims against NovaChem while defending separate regulatory inquiries.

Relationships: Potentially adverse to NovaChem India

**CLIENT-006 - Arjun Mehta** (plaintiff) - *accept*

Milder injury claimant wanting to join plaintiff-side litigation.

Arjun reports side effects and wants plaintiff representation. He is not independently adverse to existing plaintiff claimants.

Relationships: Adverse to NovaChem India

### The conflicts the grader expects to be found

| Clients | Rule |
|---|---|
| CLIENT-001 + CLIENT-003 | BCI Rule 33 |
| CLIENT-002 + CLIENT-003 | BCI Rule 33 |
| CLIENT-003 + CLIENT-004 | BCI Rule 22 |
| CLIENT-001 + CLIENT-004 | BCI Rule 33 |
| CLIENT-003 + CLIENT-005 | BCI Rule 22 |

`decline` is correct for CLIENT-003, CLIENT-004 and CLIENT-005; the other
three are `accept`. The agent is scored on reaching those decisions, on
reviewing before deciding, and on producing the conflict record.

## Privilege Review Under Litigation Pressure (`task_2`)

An in-house legal operations team must review eight documents before production. The agent needs to classify privilege correctly, spot waiver or exception risks, and decide whether each document should be produced, withheld, clawed back, or redacted.

Horizon: 34 steps. Difficulty: medium.

Ten documents. For each the agent must classify the privilege, name the
doctrine, decide produce or withhold, and identify any exception. Producing
a privileged document carries a -0.12 penalty.

### DOC-001 - External Counsel Settlement Memo

```
From: General Counsel
To: CEO
Attached is outside counsel's memo on settlement strategy for the National Consumer Disputes matter. Counsel recommends early resolution.
```

- Classification: `attorney_client`
- Doctrine: `BSA Section 132`
- Action: `withhold`
- Exception: `none`

### DOC-002 - Draft Hearing Affidavit

```
Internal litigation team draft prepared in anticipation of the Delhi High Court hearing. Contains factual chronologies, legal theories, and witness preparation notes.
```

- Classification: `work_product`
- Doctrine: `BSA Section 134`
- Action: `withhold`
- Exception: `none`

### DOC-003 - Advocate Strategy Notes

```
Handwritten advocate notes from a client briefing. The notes discuss weaknesses in the defense narrative and privileged litigation strategy.
```

- Classification: `both`
- Doctrine: `BSA Sections 132 and 134`
- Action: `withhold`
- Exception: `none`

### DOC-004 - Sales Territory Review

```
Quarterly sales performance review covering Veridex distribution targets, channel performance, and field-marketing plans.
```

- Classification: `none`
- Doctrine: `(none)`
- Action: `produce`
- Exception: `none`

### DOC-005 - Draft Toxicology Report

```
Draft expert report commissioned by counsel for upcoming trial testimony. It includes attorney comments and work product annotations.
```

- Classification: `work_product`
- Doctrine: `BSA Section 134`
- Action: `withhold`
- Exception: `none`

### DOC-006 - Deletion Directive Email

```
From: In-House Legal
To: IT Admin
Delete any internal emails mentioning 'Veridex liver toxicity' before the court issues a preservation order.
```

- Classification: `waived`
- Doctrine: `Crime-fraud exception`
- Action: `produce`
- Exception: `crime_fraud`
- Waiver event: `crime_fraud`

### DOC-007 - Public Press Release Draft

```
Draft press release quoting counsel's exact legal conclusion that Veridex is safe and compliant under all applicable regulations.
```

- Classification: `waived`
- Doctrine: `At-issue waiver`
- Action: `produce`
- Exception: `at_issue`
- Waiver event: `at_issue`

### DOC-008 - Drug Approval Certificate

```
Official Central Drugs Standard Control Organisation certificate memorializing the original approval of Veridex.
```

- Classification: `none`
- Doctrine: `(none)`
- Action: `produce`
- Exception: `none`

### DOC-009 - Joint Defense Coordination Minutes

```
Minutes from a joint-defense call between NovaChem and co-defendant counsel discussing shared defense themes, expert sequencing, and common-interest confidentiality boundaries.
```

- Classification: `both`
- Doctrine: `BSA Sections 132 and 134`
- Action: `withhold`
- Exception: `none`

### DOC-010 - Root Cause CAPA Spreadsheet

```
Operations CAPA tracker listing manufacturing deviations, corrective actions, and lot-level quality remediation owners prepared for internal compliance review.
```

- Classification: `none`
- Doctrine: `(none)`
- Action: `produce`
- Exception: `none`

Three of the ten carry no doctrine at all. That is the hole that made an
empty submission score 0.0609 - the grader read an absent field as a correct
answer. Filed as issue #1, fixed in PR #9.

## Litigation Incident Command (`task_3`)

This task simulates a legal operations incident room for a live product-liability crisis. The current scenario is pharmaceutical, but the trade-offs are intended to transfer to broader regulated litigation work. The agent must balance preservation deadlines, an ethics conflict, adversarial discovery, motion practice, regulatory response posture, and expert strategy. Solving it well requires prioritization under time pressure and avoiding hidden privilege-waiver traps.

Horizon: 24 steps. Difficulty: hard.

### EVENT-001 - Imminent Preservation Failure

Type: `spoliation_risk`. Deadline: step 6.

Auto-delete remains active on senior custodians' mailboxes while the tribunal is preparing a preservation order. A litigation hold must go out immediately.

Consequence if missed: Potential sanctions for spoliation and lost ESI.

Required action: `issue_litigation_hold`

### EVENT-002 - Emergency Sales Injunction

Type: `court_motion`. Deadline: step 9.

Plaintiffs filed for an ex parte injunction to halt Veridex sales. Counsel needs an immediate response strategy and motion work.

Consequence if missed: A sales halt could trigger revenue loss and class-action momentum.

Required action: `file_motion`

### EVENT-003 - Aggressive Discovery Request

Type: `discovery`. Deadline: step 12.

Opposing counsel demanded all communications about Veridex safety, including correspondence with advocates, under a broad Order XI request.

Consequence if missed: Producing privileged communications could waive privilege across the matter.

Required action: `respond_discovery`
Flagged as **adversarial** - the agent is being pushed toward a wrong act.

### EVENT-004 - Former Client Conflict

Type: `ethics`. Deadline: step 0.

The lead partner previously represented MedDistro in a related distribution dispute. The issue must be surfaced and resolved before strategic work continues.

Consequence if missed: Failure to address the issue risks a disqualification fight and ethics breach.

Flagged as an **ethical** issue rather than a deadline.

### EVENT-005 - Forum Consolidation Decision

Type: `coordination`. Deadline: step 18.

Parallel suits across jurisdictions are multiplying. The team needs to assess transfer and consolidation options while preserving expert strategy.

Consequence if missed: Fragmented proceedings increase cost and risk inconsistent rulings.

Required action: `file_motion`

### EVENT-006 - Regulatory Show-Cause Notice

Type: `regulatory`. Deadline: step 15.

The CDSCO issued a show-cause notice demanding immediate legal and factual response on pharmacovigilance controls, requiring parallel court-safe positioning.

Consequence if missed: Weak response can trigger plant suspension and prejudicial findings in civil litigation.

Required action: `file_motion`

Expected priority order:

EVENT-001 -> EVENT-004 -> EVENT-002 -> EVENT-003 -> EVENT-006 -> EVENT-005

