# Contributing a scenario

## Why this file exists

Every score LexCrisis reports comes from scenarios written by the same person
who wrote the graders. Asked how an artificial legal task environment can
establish construct validity, Prof. Gijs van Dijck (Maastricht) gave the
answer plainly: test it on scenarios it has not seen, and have practitioners
draft the scenarios and their answer keys, as the LegalBench study did.

Until that happens, the honest claim for this benchmark is only that a model
can learn one student's rule set. This is the intake for the repair.

## The ask, in full

**One privilege scenario. Three to ten documents. Roughly twenty minutes.**

For each document: a title, the text the agent will see, and your answer —
what it is, which provision protects it, whether you would produce or withhold
it, and whether anything defeats the claim.

You do not need to write code, install anything, or read the rest of this
repository. A filled-in JSON file, or the same information in an email, is
enough.

## The format

Copy `contrib/template_privilege.json` and fill it in.
`contrib/example_privilege.json` is a worked three-document example.

| field | what it means | allowed values |
|---|---|---|
| `doc_id` | any unique label | free text |
| `title` | what the document is called on a log | free text |
| `content` | the text the agent actually reads | free text |
| `classification` | what kind of protection, if any | `attorney_client`, `work_product`, `both`, `none`, `waived` |
| `doctrine` | the provision you would cite | free text, e.g. `BSA Section 132` |
| `action` | your call on a production request | `withhold`, `produce` |
| `exception` | what defeats an otherwise good claim | `none`, `crime_fraud`, `at_issue` |

Two known limits of that vocabulary, stated rather than hidden:

- `work_product` is a United States doctrine (FRCP 26(b)(3)) with no clean
  Indian equivalent, and the surrounding procedure — privilege logs,
  produce/withhold calls — is US-style discovery. Whether an Indian scenario
  can be expressed in it at all is an open question, and **a contribution
  saying it cannot is more useful than one that works around it.**
- Citations should be to the **Bharatiya Sakshya Adhiniyam 2023**, in force
  since 1 July 2024. The repealed Indian Evidence Act numbering still earns
  full credit, so no contribution is penalised for using it.

## Checking it before you send it

```
python validate_contribution.py contrib/your_file.json
```

The script does two things. It checks the file against the schema the graders
expect, and then it runs the real grader twice: once on your answer key, which
must score at the ceiling, and once on a submission that withholds everything,
which must not. A scenario passing both is known to **discriminate** — it can
tell a correct agent from a lazy one — which is the property a benchmark item
actually needs and the one that is easiest to get wrong.

If the script rejects your file, that is a defect in this repository's schema
or in the graders, not in your drafting. Send it anyway.

## What happens to it

Contributions land in `contrib/`, are scored separately from the built-in
scenarios, and are reported separately. If a model scores well on the built-in
scenarios and badly on contributed ones, that gap is the finding, and it will
be published as such.

You will be credited by name and affiliation unless you set `"credit": "no"`.

Open a pull request, or email
[natikchhabra22may@gmail.com](mailto:natikchhabra22may@gmail.com).

## What is already known to be wrong

Read `SCENARIOS.md` first. It carries the full contents of all three
scenarios, generated from the same Python the graders read, and it states the
open defects at the top rather than at the bottom. The headline result this
project was first published with has been retracted in this repository; see
`README.md`.
