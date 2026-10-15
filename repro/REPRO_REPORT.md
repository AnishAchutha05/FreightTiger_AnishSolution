# Three-Run Reproducibility Report

The complete pipeline was run three times on the same inputs and configuration.

- Embedding model: `BAAI/bge-small-en-v1.5`
- LLM model: `qwen2.5:7b`
- History window: 8 weeks
- Own-history threshold: 15.0%
- Peer threshold: 15.0%

## SHA-256 hashes

### weekly_analysis

- Run 1: `72bfd0fa4a6ed3bbed7f5d10ac0b9ac83c8577cbe8a7ad9702d55826cfd2d4cb`
- Run 2: `72bfd0fa4a6ed3bbed7f5d10ac0b9ac83c8577cbe8a7ad9702d55826cfd2d4cb`
- Run 3: `72bfd0fa4a6ed3bbed7f5d10ac0b9ac83c8577cbe8a7ad9702d55826cfd2d4cb`

### flagged_analysis

- Run 1: `1b45e1273a7247b6af5a1f16ac133ac9e7a43194d52d03df3bed729524d918f4`
- Run 2: `1b45e1273a7247b6af5a1f16ac133ac9e7a43194d52d03df3bed729524d918f4`
- Run 3: `1b45e1273a7247b6af5a1f16ac133ac9e7a43194d52d03df3bed729524d918f4`

### output

- Run 1: `356e4f77931df20d6ef5b7f19925c33f4b28f7be7a62d9e85d8af14becc2d466`
- Run 2: `356e4f77931df20d6ef5b7f19925c33f4b28f7be7a62d9e85d8af14becc2d466`
- Run 3: `356e4f77931df20d6ef5b7f19925c33f4b28f7be7a62d9e85d8af14becc2d466`

## Decision consistency

The following must match exactly across all runs:

- Weekly numerical analysis and row ordering
- Candidate flags and candidate classification
- Final verdicts and matched note IDs
- Submission values and row ordering

The `reason` wording is allowed to vary.

- `weekly_analysis` full-file hashes identical: Yes
- `flagged_analysis` full-file hashes identical: Yes
- `output` full-file hashes identical: Yes

## Result

PASS
