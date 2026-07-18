# Benchmark Results

Small CPU smoke benchmark for the reference implementation.

- Date: 2026-07-18
- Platform: Windows-10-10.0.26200-SP0
- Processor: Intel(R) Core(TM) Ultra 9 275HX
- Python: 3.10.9
- NumPy: 2.2.6
- Backend: NumPy CPU fallback
- Model: random 3-layer decoder, hidden_dim=96, n_heads=6, INT4 weights
- Decode length: 48 generated tokens per prompt

| Benchmark | Mode | Cache | Batch | Tokens | Avg seconds | Tok/s | Accept rate | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Greedy single decode | greedy | plain | 1 | 48 | 0.281 | 170.9 | - | single prompt |
| Greedy single decode | greedy | paged | 1 | 48 | 0.287 | 167.1 | - | single prompt |
| Prompt lookup speculation | lookup | paged | 1 | 48 | 0.199 | 241.0 | 94.7% | single prompt |
| N-gram speculation | ngram | paged | 1 | 48 | 0.188 | 255.4 | 0.0% | single prompt |
| Neural draft speculation | model | paged | 1 | 48 | 0.517 | 92.9 | 0.5% | single prompt |
| Greedy batched decode | greedy | plain | 4 | 192 | 0.377 | 509.5 | - | aggregate throughput |

These measurements are best used for regression tracking inside this
repository. They are not production throughput claims.
