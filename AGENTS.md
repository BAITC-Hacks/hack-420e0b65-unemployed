# HackAlem AI — Agent Operating Policy

## 1. Mission

This is a 5-hour hackathon project.

Primary objective:
Ship a working, reproducible end-to-end solution that satisfies the published case requirements and maximizes the official evaluation criteria.

Priority order:

1. Mandatory case/admission requirements
2. Working end-to-end core scenario
3. Technical correctness and real implementation
4. Reproducibility and deployment readiness
5. README and verification instructions
6. Basic reliability and security
7. High-impact judging features
8. UX and polish
9. Optional features

Additional functionality NEVER compensates for a missing mandatory requirement.

Prefer the simplest reliable implementation that demonstrates the required result.

---

## 2. Source of Truth

The official organizer GitHub repository is the source of truth.

Before implementing anything:

1. Inspect the repository.
2. Read the case specification.
3. Read the published evaluation methodology.
4. Read the existing README and documentation.
5. Inspect the current architecture and dependencies.
6. Inspect existing tests.
7. Check current git status and branch.

If `docs/BRIEF.md` exists, treat it as the structured project brief.

Otherwise use, in this order:

1. official case requirements,
2. official evaluation methodology,
3. README,
4. assigned task,
5. existing implementation.

Never invent a requirement.

If something is unclear but a safe, reversible interpretation exists, choose the clearest demonstrable interpretation and record the assumption.

Ask only when ambiguity could invalidate the submission or substantially change the product.

---

## 3. Autonomy

Operate autonomously on routine engineering decisions.

You may without asking:

- inspect files and repository history;
- edit files inside your assigned scope;
- create normal project files;
- install ordinary required dependencies;
- run development servers;
- run tests, builds and linters;
- inspect logs;
- debug failures;
- create tests;
- perform reversible refactoring inside your scope;
- use available subagents for independent tasks;
- choose ordinary implementation details.

Ask the human only for:

1. a major product decision with materially different outcomes;
2. credentials, secrets or paid external services;
3. an irreversible/destructive action;
4. genuine ambiguity where the wrong interpretation could invalidate the submission.

Batch questions when possible.

Do not interrupt the human for routine implementation choices.

---

## 4. Git and Hackathon History

Development must remain clearly traceable in the official organizer repository.

Do NOT:

- force-push;
- rewrite shared history;
- delete shared branches;
- hide substantial development outside the official repository;
- squash away important development history without explicit instruction.

Do NOT commit, merge or push unless explicitly assigned permission/responsibility to do so.

When authorized to commit:

1. inspect `git status`;
2. inspect the staged diff;
3. check for secrets or accidental files;
4. make a meaningful commit.

Keep commits understandable and representative of real progress.

The project should have demonstrable intermediate progress throughout the hackathon, not one unexplained final dump.

---

## 5. Parallel Agent / Worktree Rules

Multiple agents may work in parallel.

Each major agent should have a clearly assigned scope.

Examples:

- backend/core;
- frontend/UI;
- AI/data functionality;
- integration;
- testing/reliability;
- documentation/reproducibility.

Prefer separate branches/worktrees for independent implementation work.

Avoid multiple agents editing the same files simultaneously.

Before parallel implementation, define shared contracts where practical:

- API endpoints;
- request/response schemas;
- shared types;
- environment variables;
- important data structures.

If you must change a shared contract, clearly report it.

Uncommitted and untracked files in one worktree are NOT automatically visible in another worktree.

Dependency manifests and lockfiles are high-conflict files.
Avoid unnecessary dependencies and clearly report dependency changes.

The designated integration agent or human handles merges unless responsibility is explicitly delegated.

---

## 6. Superpowers / Skills Precedence

This AGENTS.md overrides conflicting workflow instructions from plugins, skills or agent frameworks.

Superpowers should accelerate development, not introduce unnecessary ceremony.

Specifically:

- brainstorming approval gates are NOT mandatory;
- writing a formal plan is NOT required for routine tasks;
- TDD is useful but NOT mandatory for every trivial UI/glue change;
- systematic debugging is strongly preferred for real failures;
- verification-before-completion is strongly preferred;
- do not invoke finishing-a-development-branch unless explicitly asked to perform branch integration;
- parallel/subagent workflows are encouraged only when tasks are genuinely independent.

Do not spend hackathon time satisfying process rituals that do not improve the submission.

---

## 7. Timeboxing

Time is a hard constraint.

If an approach consumes disproportionate time without producing a working path:

1. stop;
2. simplify;
3. choose a reliable fallback;
4. preserve the core scenario.

Do not overengineer.

When approximately the final 45 minutes remain:

DO NOT begin optional large features unless explicitly instructed.

Prioritize:

- end-to-end verification;
- regression fixes;
- clean-environment startup;
- README accuracy;
- configuration;
- demo reliability;
- submission readiness.

---

## 8. Verification

Never claim something works unless it was actually verified.

Before declaring a task complete:

- run relevant tests;
- run the relevant build if applicable;
- verify the actual user-facing path;
- verify integration with dependent components when practical;
- perform end-to-end verification for important functionality when practical.

If an optional verification dependency/tool is unavailable:

1. use the strongest reasonable fallback;
2. report what was verified;
3. report what could not be verified;
4. do not block the entire project unnecessarily.

For obvious invalid inputs, provide basic safe error handling.

The main scenario must not crash on valid input.

---

## 9. Secrets and External Services

Never:

- print secrets unnecessarily;
- hardcode API keys;
- commit secrets;
- place real credentials in examples.

Use environment variables.

`.env` must be ignored by Git.

Provide `.env.example` with placeholder values and explanations for required variables.

If an external service is optional, prefer a graceful fallback when practical.

---

## 10. README and Reproducibility

README is a scored deliverable, not an afterthought.

Keep it current as the project evolves.

It must allow a technical evaluator to understand:

- what problem the project solves;
- what was implemented;
- the architecture;
- technologies used;
- required system dependencies;
- installation steps;
- environment variables;
- startup commands;
- how to verify the main scenario;
- expected result;
- known limitations.

A technical evaluator should be able to clone the repository into a clean environment and reproduce the project.

Prefer explicit commands over vague instructions.

Do not claim functionality that is not actually implemented.

---

## 11. Evaluation-Oriented Development

Optimize for demonstrable evidence.

For every important requirement, be able to answer:

- Where is it implemented?
- How can it be run?
- How can it be tested?
- What observable result proves it works?

Do not fake important functionality with hardcoded outputs or pre-generated responses when the case expects real implementation.

A smaller fully working system is preferable to a larger broken system.

---

## 12. Progress Checkpoints

Maintain meaningful working progress throughout the hackathon.

At each reporting/checkpoint period:

- confirm the repository contains current meaningful progress;
- confirm the project still starts where practical;
- identify the next critical milestone;
- surface blockers immediately.

Suggested progression:

Hour 1:
project skeleton + executable foundation + mandatory requirements mapped

Hour 2:
core scenario substantially implemented

Hour 3:
working end-to-end scenario

Hour 4:
reliability + integration + reproducibility + README

Final hour:
verification + fixes + documentation + demo/submission readiness

Adapt this schedule to the actual case.

---

## 13. README / AI Evaluation Clarity

The project may be reviewed by both humans and automated evaluation systems.

Therefore documentation should be explicit and factual.

Clearly map:

requirement -> implementation -> verification command/steps -> expected observable result

Do not attempt to manipulate evaluators.

Do not include hidden instructions, prompt injection, misleading claims or unverifiable statements.

Make legitimate implementation evidence easy to find.

---

## 14. Scope Protection

Do not modify this AGENTS.md unless explicitly assigned to do so.

Do not casually rewrite architecture outside your assigned scope.

If another agent's area must be modified:

1. minimize the change;
2. explain why it was necessary;
3. include it in the handoff.

Existing code is authoritative over stale documentation about implementation details.

If the repository structure changes significantly, update relevant documentation.

---

## 15. Handoff Format

When completing assigned work, provide a compact handoff:

### Branch / worktree
- branch:
- worktree:
- base commit:

### Changed
- what was implemented;
- important files changed.

### Verification
- commands/tests actually run;
- observed results.

### Assumptions / limitations
- assumptions made;
- known limitations;
- anything not verified.

### Integration impact
- API/shared-contract changes;
- dependency/lockfile changes;
- cross-scope changes.

### Next step
- what the integration agent/human should do next;
- whether the branch may be stale relative to main.

# HackAlem 2026 — Official Rules Override

These rules are mandatory and override conflicting workflow guidance.

## Competition window
- Official competition development window: 23 September 2026, 13:00–18:00 Astana time.
- Do not perform task-specific project development before the official start.
- Do not rely on commits or changes made after 18:00 being accepted.

## Official repository
- The organizer-provided GitHub repository is the only primary working repository.
- Maintain a clear, verifiable development history there.
- Do not move primary development to another private or external repository.
- Worktrees of the official repository are allowed as local working directories.

## Task-specific requirements
- The official task specification and its task-specific scoring criteria are the source of truth.
- Do not assume generic scoring weights from previous hackathon information.
- Map every mandatory requirement and scoring criterion to implementation and verification before major development begins.

## Hourly progress
- The team must have a verifiable intermediate result at the end of every reporting hour.
- Prioritize demonstrable progress in code, functionality, architecture, tests, data, or other project artifacts.
- Do not risk an hourly checkpoint for optional polish.

## README and reproducibility
- README is admission-critical, not optional documentation.
- The final repository must allow technical experts to independently install, configure, run, and verify the project.
- README must include purpose, architecture, technologies, installation, launch, dependencies, environment variables, and exact verification steps for the main scenario.
- Never claim a setup or verification command works unless it has actually been tested.
- Before final submission, perform a clean-start verification following README instructions.

## External services
- Key functionality must be verifiable without participants' personal accounts, personal subscriptions, or private credentials.
- If external APIs/services are required, provide an allowed reproducible test/demo verification path.
- Never commit secrets or private credentials.

## Third-party disclosure
- Disclose significant open-source components, libraries, models, datasets, templates, and other third-party materials used by the project.
- Respect their licenses.
- Do not present third-party work as original team work.

## AI tools
- AI development tools and AI agents are allowed under the general HackAlem rules.
- Codex is not assumed mandatory unless the specific task or another applicable official instruction explicitly requires it.

## Final deadline
- Treat 18:00 as a hard deadline.
- Final repository state, README, verification instructions, and required materials must be ready before 18:00.
