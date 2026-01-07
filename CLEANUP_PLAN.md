# Cleanup & Claude Hooks Plan

**Goal**: Clean up tracked folders, set up Claude Code hooks for enforcing 100% quality gates.

---

## Current State Analysis

### Folder Assessment

| Folder | Size | Tracked | Status | Action |
|--------|------|---------|--------|--------|
| `src/` | 44K | Yes | **NEEDED** | Keep - package code |
| `scripts/` | 768K | Yes | **NEEDED** | Keep - pipeline scripts |
| `tests/` | 144K | Yes | **NEEDED** | Keep - test suite |
| `alembic/` | 84K | Yes | **NEEDED** | Keep - migrations |
| `docs/` | 340K | Yes | Review | Keep useful docs, remove stale |
| `.beads/` | 1.3M | Yes | **REMOVE** | Runtime files, should not be tracked |
| `db/` | 84K | Yes | **REMOVE** | Old SurrealDB code, unused |
| `checkpoints/` | 9.2G | Partial | **REMOVE** | Binary model files (*.pt) |
| `eval_results/` | 120K | Yes | **REMOVE** | Evaluation artifacts |
| `models/` | 7.2M | Yes | **REMOVE** | Model files |
| `results/` | 24K | Yes | **REMOVE** | Old results |
| `archive/` | 3.1M | No | Already ignored | Can delete if exists |
| `backups/` | 6.8G | No | Already ignored | Keep locally |
| `data/` | 73G | No | Already ignored | Keep locally |
| `htmlcov/` | 208K | No | Should ignore | Coverage reports |

### CLI Status

The CLI was removed because it was a 968-line copy of `scripts/onboard_data.py`.

**Current approach**: Run scripts directly:
```bash
uv run python scripts/onboard_data.py --status
# or
make status
```

**Alternative**: If you want a proper CLI, we can create a thin wrapper in the package.

---

## Iteration Plan

### Iteration 9: Cleanup Tracked Folders

**Goal**: Remove unnecessary tracked files and update .gitignore

**Tasks**:
1. Remove tracked files that shouldn't be committed:
   - `.beads/*` (except README)
   - `db/` (old SurrealDB code)
   - `checkpoints/*.pt` (model binaries)
   - `eval_results/*`
   - `models/*`
   - `results/*`
   - `.coverage`

2. Update `.gitignore` to prevent re-addition:
   ```
   # Generated files
   .coverage
   htmlcov/

   # Model files
   *.pt
   *.onnx
   *.safetensors

   # Evaluation artifacts
   eval_results/
   models/
   results/

   # Old/unused
   db/
   .beads/
   ```

3. Clean up `docs/` - keep only current documentation

**Verification Contract**:
```bash
# MUST have fewer tracked files
git ls-files | wc -l  # Should be < 50

# MUST NOT track these patterns
git ls-files | grep -E "\.pt$|eval_results|models/|db/|\.beads/" && echo "FAIL" || echo "PASS"

# make check MUST still pass
make check
```

**Pass/Fail Criteria**:
- PASS: Tracked files reduced, make check passes
- FAIL: Any quality gate fails

---

### Iteration 10: Claude Code Hooks Setup

**Goal**: Enforce 100% quality gates via Claude Code hooks

**Tasks**:

1. Create `.claude/settings.json` (committed, shared):
   ```json
   {
     "hooks": {
       "PostToolUse": [
         {
           "matcher": "Edit|Write",
           "hooks": [
             {
               "type": "command",
               "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/post-edit.sh",
               "timeout": 30
             }
           ]
         }
       ],
       "Stop": [
         {
           "hooks": [
             {
               "type": "command",
               "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/pre-stop.sh",
               "timeout": 120
             }
           ]
         }
       ]
     }
   }
   ```

2. Create `.claude/hooks/` directory with scripts:

   **`.claude/hooks/post-edit.sh`** - Quick check after each edit:
   ```bash
   #!/bin/bash
   # Quick lint check after file edits
   cd "$CLAUDE_PROJECT_DIR"

   # Only check Python files in src/ or tests/
   file_path=$(cat | jq -r '.tool_input.file_path // empty' 2>/dev/null)
   if [[ "$file_path" =~ ^.*(src|tests)/.*\.py$ ]]; then
       uv run ruff check "$file_path" --fix --quiet 2>/dev/null || true
       uv run ruff format "$file_path" --quiet 2>/dev/null || true
   fi
   exit 0
   ```

   **`.claude/hooks/pre-stop.sh`** - Full check before Claude stops:
   ```bash
   #!/bin/bash
   # Run full quality gates before Claude finishes
   set -e
   cd "$CLAUDE_PROJECT_DIR"

   echo "Running quality gates..."

   # Type checking
   if ! uv run mypy src/ --strict --no-error-summary 2>/dev/null; then
       echo "❌ Type check failed" >&2
       exit 2
   fi

   # Tests with coverage
   if ! uv run pytest tests/unit/ -q --tb=short 2>/dev/null; then
       echo "❌ Tests failed" >&2
       exit 2
   fi

   echo "✅ All quality gates passed"
   exit 0
   ```

3. Make hooks executable:
   ```bash
   chmod +x .claude/hooks/*.sh
   ```

4. Update `.gitignore` to NOT ignore `.claude/settings.json` and `.claude/hooks/`

**Verification Contract**:
```bash
# MUST have settings.json
test -f .claude/settings.json && echo "PASS: settings.json exists"

# MUST have hooks directory
test -d .claude/hooks && echo "PASS: hooks dir exists"

# Hooks MUST be executable
test -x .claude/hooks/post-edit.sh && echo "PASS: post-edit.sh executable"
test -x .claude/hooks/pre-stop.sh && echo "PASS: pre-stop.sh executable"

# make check MUST pass (hooks shouldn't break anything)
make check

# Claude hooks command MUST show hooks
# (run interactively): /hooks
```

**Pass/Fail Criteria**:
- PASS: All hooks exist, are executable, make check passes
- FAIL: Missing hooks or quality gate failures

---

### Iteration 11: Optional CLI Package Entry Point

**Goal**: Add clean CLI entry point (optional)

**Tasks**:

1. Create `src/rl_emails/cli/__init__.py`:
   ```python
   """CLI entry point."""
   from __future__ import annotations

   import subprocess
   import sys
   from pathlib import Path


   def main() -> None:
       """Run the pipeline via scripts/onboard_data.py."""
       scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
       script = scripts_dir / "onboard_data.py"

       if not script.exists():
           print(f"Error: {script} not found")
           sys.exit(1)

       result = subprocess.run([sys.executable, str(script)] + sys.argv[1:])
       sys.exit(result.returncode)
   ```

2. Update `pyproject.toml`:
   ```toml
   [project.scripts]
   rl-emails = "rl_emails.cli:main"
   ```

3. Add tests for CLI module (maintain 100% coverage)

**Verification Contract**:
```bash
# CLI MUST be installable
uv run rl-emails --help 2>/dev/null | grep -q "onboard" && echo "PASS"

# Coverage MUST remain 100%
make coverage
```

**Pass/Fail Criteria**:
- PASS: CLI works, 100% coverage maintained
- FAIL: CLI fails or coverage drops

---

## Summary

| Iteration | Goal | Key Deliverable |
|-----------|------|-----------------|
| 9 | Cleanup | Remove ~30 tracked files |
| 10 | Hooks | Enforce quality via Claude hooks |
| 11 | CLI | Optional - clean entry point |

**Priority**: Iteration 9 and 10 are essential. Iteration 11 is optional.

---

## Commands Reference

After completion:

```bash
# Development
make check              # All quality gates
make coverage           # 100% coverage check

# Pipeline
make run                # Run pipeline
make status             # Check status
rl-emails --status      # (after Iteration 11)

# Claude hooks (in Claude session)
/hooks                  # View configured hooks
```
