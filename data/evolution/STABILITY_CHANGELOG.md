# Evolution Stability Changelog

Date: 2026-02-16
Scope: hard task stability (`hard_calc_parser`, `hard_concurrent_pool`)

## Final Stable Configuration

File: `evolve.py`

1. Increased task-specific validation feedback retries:
   - `hard_calc_parser`: `3`
   - `hard_concurrent_pool`: `3`

2. Added stronger benchmark prompt constraints for hard tasks:
   - `hard_calc_parser`: minus-sign handling, precedence discipline, end-of-parse token check, self-check examples.
   - `hard_concurrent_pool`: explicit worker execution requirement, future completion semantics, graceful shutdown/join, comparable `PriorityQueue` tuple rule.

3. Added error-aware retry prompt guidance:
   - `hard_calc_parser`: targeted fixes for invalid number parse and assertion failures.
   - `hard_concurrent_pool`: targeted fixes for timeout hangs and `PriorityQueue` comparability (`NoneType` comparison) failures.

## Evidence Windows

### Recovery Window (after pool-focused prompt tightening)
- Window: `R374–R381`
- Overall:
  - avg round pass ratio: `0.9844`
  - std round pass ratio: `0.0413`
  - full-pass rounds: `7/8`
- Key tasks:
  - `hard_calc_parser`: `1.0000`
  - `hard_concurrent_pool`: `0.8571`
- Result: both hard tasks >= `0.80` (PASS)

### Confirmation Window (stability acceptance)
- Window: `R382–R391`
- Overall:
  - avg round pass ratio: `0.9875`
  - std round pass ratio: `0.0375`
  - full-pass rounds: `9/10`
- Key tasks:
  - `hard_calc_parser`: `1.0000`
  - `hard_concurrent_pool`: `0.8000`
- Result: both hard tasks >= `0.80` (PASS)

## Current Snapshot
- Round: `391`
- Overall pass rate: `0.734896` (`815/1109`)
- Best score: `8.75` at round `220`

## Acceptance Decision
- Stability objective met for the current acceptance criterion:
  - `hard_calc_parser >= 0.80`
  - `hard_concurrent_pool >= 0.80`
  - sustained in a dedicated confirmation window.

## Revalidation Command

```bash
/Users/jiangshengyu/Documents/program/python/Python_Coding_Agent/.venv/bin/python evolve.py --resume --rounds 10
```

Compare the newest 10-round window against `R382–R391` using the same metrics:
- task pass rate,
- fails/flips,
- avg/std round pass ratio,
- full-pass rounds.
