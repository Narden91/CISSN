# Agent learnings

Each entry: the concrete failure, then the rule that prevents it.

## PowerShell run blocks

- **Failure:** `$flag = if ($x) { @('--revin') } else { @() }` followed by `@flag` passed
  `- - r e v i n` to the program. An `if` expression unrolls a one-element array into a
  string, and splatting a string passes each character as an argument.
  **Rule:** assign arrays directly (`$flag = @(); if ($x) { $flag = @('--revin') }`) before
  splatting.
- **Failure:** `--ablations a,b,c` split into separate arguments. An unquoted comma list is a
  PowerShell array.
  **Rule:** quote comma lists, or use a value that needs no comma.
- **Failure:** run commands were written into `experiments.md` after a parse check only, and the
  first launch failed.
  **Rule:** before writing a launch command into a doc, run it through the target script's
  real argument parser with the real PowerShell block (override `uv` to call a parse-only
  wrapper). Check every skip-check wildcard against the runner's real name builder.
