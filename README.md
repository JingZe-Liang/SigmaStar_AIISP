# Report Workflow Project

This directory is a standardized report-writing project.

## How to Use

1. Put problem files into `input/problems/`.
2. Put reference materials into `input/references/`.
3. Put template files into `input/template/` if any.
4. Edit `task_config.yaml`.
5. Invoke the `report-workflow` skill again.

## Directory Rules

- `input/` is read-only for the agent.
- `work/` stores task inventory, notes, drafts, code, assets, and checks.
- `output/` stores final deliverables only.

## Expected Workflow

The agent will:
1. Read `task_config.yaml`.
2. Read problems.
3. Read references.
4. Read template if provided.
5. Create task inventory.
6. Record notes, pitfalls, ambiguities, and assumptions.
7. Draft answers.
8. Review content correctness and revise the draft.
9. Apply layout, formula rendering, and template placement.
10. Generate final report.
11. Check final correctness and formatting.
