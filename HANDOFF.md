# Mol-1 MLX port: handoff

Last updated 2026-10-28 (hybrid, worker, fusion). Machine: M3 Max, 48 GB, macOS, MLX 0.32.2, torch 2.12.0 (MPS). Claims
are tagged VERIFIED (a command ran and its output was read), INFERRED (arithmetic or code
reading, not measured) or NOT DONE.

> **Status: the hybrid works end to end and is faster at every size measured, by an amount that depends
> heavily on N and on how many recycle cycles you run.** `bin/mol1-mlx pred ...` takes `mol pred`'s flags
> and runs the recycling trunk (recycle plumbing, template embedder, MSA module, 48-block stack) in MLX in a
> worker process; Mol-1's own torch code does everything else. It reproduces torch's confidences to three
> decimals (3 seeds). Warm, N=97, 4 cycles / 20 steps / 1 sample: 4.53 s torch vs 3.57 s (1.27x).
> Trunk only, 4 cycles: ~1.1-1.5x at N=97-128, ~1.5x at N=192, ~1.6x at N=256, ~2.0-2.1x at N=320-384 (tables below; torch's
> baseline is noisy). Diffusion and the confidence head are still torch. FlashIPA was not done (Mol-1 has no
> IPA); see Scope.
>
> **Nothing here is committed.** `git init` was run but there are no commits. `refs/` (1.7 GB)
> and `weights/` (2.1 GB) are gitignored and regenerable (see "Reproducing"). The Mol-1
> checkout at `~/Mol-1` also carries its own uncommitted work; this port imports `mol` only
> for featurization and for the reference captures.

## Decisions made by the user

- **fp32 only. No half precision (fp16 or bf16).** Decided 2026-10-28. **This holds for the whole pipeline on this Mac**: the MLX trunk is fp32, and Mol-1's `-d bf16` default (the CLI flag) only enables CUDA autocast (`mol/runner/inference.py:93-95,196-201`: `autocast("cuda") if cuda else nullcontext()`; its own log says "dtype=bf16 applies through CUDA autocast only; mps runs fp32"), so diffusion and the confidence head are fp32 on MPS too. An earlier version of this file called the torch side "bf16": wrong. It was measured (see "Half
  precision: measured, not pursued") and is neither recommended nor planned. `mol_mlx/precision.py`
  remains as an opt-in switch, default off; every test and every number below that is not in that
  section is fp32.

## Scope, as read

"Port to MLX to speed up inference, with Metal kernels for the CUDA-style kernels." Taken as:
Mol-1 inference in MLX, with hand-written Metal where MLX's native ops lose, replacing the
cuEquivariance and Triton surfaces. FlashIPA (arXiv 2505.11580) was **not** ported: Mol-1 has no
invariant point attention (AF3-class; the diffusion module replaced IPA), and FlashIPA
factorizes the pair representation, an approximation that requires retraining. Parked. **Open
question for the user:** where IPA does exist in their stack (`genie3`'s PyTorch module, or
BindCraft2's JAX AF2 structure module), which one, and are they prepared to retrain (FlashIPA's
own evaluations retrain FoldFlow and RNA-FrameFlow for ~20 h each)?

## Done and VERIFIED (134 tests pass: `PYTHONPATH=. pytest tests -q`)

| Component | Evidence |
| --- | --- |
| LayerNorm, Linear, Transition (SwiGLU) | vs real checkpoint, N=48/72/97; plus random weights vs torch |
| Triangle multiplication (out/in) | vs real checkpoint, blocks 0/8/47, 3 sizes |
| Triangle attention (start/end), dense and Metal | vs real checkpoint, blocks 8/47, 3 sizes, both paths |
| `AttentionPairBias` | **random non-zero weights vs torch only** (see finding 2) |
| PairMixer and Pairformer blocks | every block, N=48: ~1e-7 relative, no growth with depth |
| 48-block `PairformerStack` + both width adapters | 2 recycle cycles x N=48/72/97: s ~7e-7, z ~1.5e-6 relative |
| `DistogramHead` | N=48/72/97: 3e-7 to 8e-7 relative |
| Recycle loop + template embedder + MSA module (3 cycles, replaying the real model's per-cycle MSA samples) | z 3e-6 relative, distogram 1-2e-6, 0% bin flips at N=48 and 97 (`tests/test_recycle_vs_refs.py`) |
| Worker process (`mol_mlx/worker.py`) | output matches the in-process Trunk to <1e-5; reused across requests; errors raised not swallowed; killed mid-run raises and restarts (`tests/test_worker.py`) |
| `mol1-mlx pred` vs `mol pred`, real CLI, N=97, 3 seeds | pLDDT / pTM / ipTM / ranking score / gPDE identical to 3 decimals (pLDDT 71.942 vs 71.941) |
| Weight conversion | 4048 keys / 576.79M params, matches Mol-1's own count; strict use accounting |
| Metal flash kernels, triangle attention: v1 (one query/thread) and **v2 (simdgroup matrix tiles, default)** | both vs dense <1e-5 incl. N=5, 53, 97, 130, 14 heads, mask exclusion across the pad boundary, large logits |

References are captured from the **real checkpoint on torch CPU** (`scripts/capture_refs.py`),
one module at a time, so the GPU only ever holds MLX tensors during a test.

## Findings that change decisions

1. **MLX's fused `scaled_dot_product_attention` does not fuse at triangle-attention shapes.**
   head_dim is 32 everywhere in Mol-1; MLX 0.32 falls back to the naive path: bit-identical
   output, identical timing, and it materializes the `[B,H,N,N]` scores (a 600-token call is
   ~8.6 GB of scores). `mol_mlx/tri_attn_metal.py` is the flash-style replacement.
   Under a hard 6 GiB cap per process, the kernel ran every N through 512; dense died at 512,
   MLX-fused at 448. At N=448: 58 ms vs 180 ms dense (3.1x); at N=384, 37 ms vs 152 ms fused.
2. **The single track is numerically dead in this checkpoint.** `attention_pair_bias` output
   weights are ~1e-32 in all 48 blocks; `single_transition` is live only in blocks 40 and 43
   (weights up to 0.85 and 0.42). 46 of 48 blocks leave `s` unchanged. Consequence for
   validation: matching captured activations proves nothing for `attention_pair_bias` (0 vs 0),
   hence the random-weight tests. Consequence for speed: it can be skipped exactly; NOT DONE.
3. **`s` reaches ~3e5 after block 43.** fp16 (max 65,504) would overflow. Recorded in case half
   precision is ever revisited; not relevant to the fp32 path.
4. **MLX's peak-memory counter is unreliable** here: a known 64 MiB allocation read as 0.0 MiB
   while 256 MiB read correctly. All memory claims use hard-cap survival (largest N that
   finishes under a cap) or the watchdog's sampled maximum, labelled as a proxy.
5. **Silent-miss hazard, fixed.** LayerNorm scale/offset are legitimately optional, so a
   mistyped weight prefix looked identical to "no affine" and changed the math silently.
   `Weights` now records reads, and `PairformerStack` refuses to build unless all 2610 trunk
   keys are consumed.

## Benchmarks (VERIFIED; fp32; one 48-block stack pass; fresh process per row)

The torch column is the stack **called the way `mol pred` calls it** (eval + no_grad turns
`inplace_safe` on, `chunk_size=1024`, transition chunks 128; mol/model/mol.py:991-992 and the config
defaults). An earlier version used `inplace_safe=False, chunk_size=None`, which is not what production
runs, and overstated the speedup. Torch is one measurement per N (~6% noise).

**Run-to-run noise is real; treat differences under ~10% as noise.** MLX fp32 at N=256 measured
4.78, 4.40 and 4.52 s with the v2 kernel and 5.17, 5.09 and 5.29 s with v1; at N=128, 1.09-1.32 s (v2)
and 1.15-1.28 s (v1). The same op repeated in a block profile varied ~10% (pair transition 23.5 vs
26.0 ms, identical code).

| N | MLX fp32, v1 kernel | MLX fp32, v2 kernel (default) | MLX dense fp32 | torch/MPS (production call) | v2 vs torch |
| --- | --- | --- | --- | --- | --- |
| 128 | 1.15-1.28 s | 1.09-1.32 s | 1.23 s | 1.40 s | ~1.05-1.3x (noise-limited) |
| 192 | 2.79 s | - | 3.04 s | 3.72 s | - |
| 250 | 4.85 s | 4.31 s | - | - | - |
| 256 | 5.09-5.29 s | 4.40-4.78 s | 6.44 s | 7.21 s | ~1.5-1.6x |
| 384 | 13.49-13.58 s | 11.79 s | **killed at 8.39 GiB (8 GiB cap)** | - | - |

At N=384 the Metal path completes the stack in 5.6-5.8 GiB where dense MLX is killed by the same
8 GiB cap, so at stack level the kernel is what keeps it in memory (at N<=256 held memory is about
equal: weights, 1.2 GB for the stack, and allocator cache dominate). That is relative to my cap, not
a machine limit. Memory figures are the watchdog's sampled maximum (a proxy). No torch comparison
above 256 (that regime needs ~29 GB under torch; deliberately not run).

The stack is ~28 TFLOP at N=256 (INFERRED from FLOP counting), so 4.4-4.8 s is ~5.8-6.4 TFLOP/s,
against the 10-11 TFLOP/s that isolated fp32 matmuls reach on this GPU (measured,
`bench/matmul_dtypes.py`). The gap is memory-bound elementwise passes and layout copies.

## Half precision: measured, NOT pursued (the user decided fp32 only)

Kept as reference data only. All of it is one pass from real cycle-1 inputs (no recycling), on the v2
kernel, against the torch CPU fp32 reference (`bench/bf16_accuracy.py`, `bench/matmul_dtypes.py`).

- Raw matmul at the stack's shapes: bf16/fp16 are 1.10-1.19x fp32 (this GPU has no tensor-core-style
  2x; fp32 already reaches ~10 TFLOP/s). Whole stack: ~1.13x at N=256 (4.40-4.52 s -> 3.88-3.98 s);
  at N=128 0.92 s vs 1.32 s, noise-limited.
- Accuracy: fp32 MLX z ~1e-6; **bf16** z 5.7e-3 to 7.8e-3, 0.38-0.66% of token pairs change their
  likeliest distance bin, max change in P(contact<8 A) 0.012-0.014; **fp16** z 8.2e-4 to 8.7e-4,
  0-0.12% bin changes, max 0.0016-0.0026.
- Constraints found: the single track must stay fp32 (`s` reaches ~3e5; fp16 gives NaN there);
  a prior Protenix port saw bf16 quantize pLDDT; recycling error and fp16's overflow margin on other
  inputs were never tested.

## Where one block's time goes (VERIFIED, N=256, fp32, synced per op; `bench/block_profile.py`)

Pairformer block (40 of 48 blocks, 84% of the stack), ms per block, before and after the v2 kernel:

| op | v1 kernel | v2 kernel | share now |
| --- | --- | --- | --- |
| triangle multiplication x2 | 41.8 | 40.6 | 42% |
| triangle attention x2 | 42.7 | 27.7 | 29% |
| pair transition | 23.5 | 26.0 | 27% |
| single track | 2.1 | 2.3 | 2% |
| **block** | **110.1** | **96.7** | |

Triangle multiplication and the transitions are about half to 60% of the matmul roofline (INFERRED:
~105 and ~161 GFLOP per call at c_z=320 against ~20 and ~26 ms measured, at 10-11 TFLOP/s). The v1
attention kernel ran at ~1.5-2 TFLOP/s against a measured scalar-FMA peak of 9.4 TFLOP/s
(`bench/scalar_fma_peak.py`), so it was not FMA-rate-bound; the ablation below located the cost.

## Kernel v2: simdgroup matrix tiles, fp32 (DONE, VERIFIED)

**Ablate before restructuring.** Removing one component at a time from v1 (`bench/kernel_ablation.py`,
N=256, 11.0 ms baseline): the per-key bias loads (1.01x), both exps (1.00x) and the accumulator
rescale (0.98x) changed NOTHING. Removing the QK dot gave 0.67x and removing the PV accumulate 0.36x.
So the earlier guesses (exp, rescale, bias, scalar FMA rate) were wrong; the cost is the two 32-wide
inner loops, and the best reading (not proven) is register pressure from a 32-float query plus a
32-float accumulator per thread with no operand reuse.

**Design** (`mol_mlx/tri_attn_mma.py`): a simdgroup owns 8 queries; Q and the output accumulator are
8 matrix tiles; S = 4 chained 8x8x8 MMAs over the head dim; online softmax per row with two XOR
shuffles; P@V as 4 more MMAs. Lane layout was MEASURED, not assumed (`bench/simdgroup_probe.py`): each
lane holds 2 adjacent elements of one row, row = ((l>>1)&3)|(((l>>4)&1)<<2), the 4 lanes of a row are
{l, l^1, l^8, l^9}. Masking uses a large finite negative (-1e30), never -INFINITY, because Metal's
fast-math may assume no infinities. N is zero-padded to a multiple of 8 (a copy, only when needed);
padded keys are masked in-kernel.

**Kernel-only results** (fp32, `bench/tri_attn_v2_bench.py`, error vs dense 7e-7 to 1.4e-6):

| N | v1 | v2 | v2 vs v1 | v2 vs dense MLX | v2 attention TFLOP/s |
| --- | --- | --- | --- | --- | --- |
| 128 | 1.76 ms | 0.74 ms | 2.4x | 3.2x | 3.6 |
| 250 (not a multiple of 8) | 10.74 ms | 5.59 ms | 1.9x | 2.9x | 3.6 |
| 256 | 11.29 ms | 3.68 ms | 3.1x | 4.5x | 5.8 |
| 384 | 37.35 ms | 11.43 ms | 3.3x | 9.1x | 6.3 |

The N=250 vs N=256 gap (5.59 vs 3.68 ms) is the padding copy and output slice.

**Whole-stack results** (fp32, one 48-block pass; ranges are repeat runs, see the noise note above):
N=128 1.15-1.28 -> 1.09-1.32 s (within noise); N=250 4.85 -> 4.31 s; N=256 5.09-5.29 -> 4.40-4.78 s
(~1.1x); N=384 13.49 -> 11.79 s (~1.14x), held memory 5.79 -> 5.63 GiB. Against torch/MPS called the
way `mol pred` calls it (7.21 s at N=256): roughly 1.5-1.6x. The padding copy visible in the kernel-only
table is not visible at stack level (N=250 ran no slower than interpolating from N=256), so removing
it is not worth doing.

**Correction to my earlier estimate.** I said the new kernel would save "~1 s of the 4.5 s stack".
The kernel did hit the ~3x I predicted (3.1x), but the stack only gained ~0.5 s (11%), because I had
attributed the WHOLE triangle-attention op to the kernel; the kernel was only about half of that op's
~21 ms (the rest is five projections, layer norm, gating and layout copies). Profile after v2
(fp32, N=256, Pairformer block, 96.7 ms, was 110.1): tri-mul x2 40.6 ms (42%), tri-attn x2 27.7 ms
(29%, was 42.7), pair transition 26.0 ms (27%), single track 2.3 ms.

## The recycling trunk, the hybrid, and the process finding (DONE, VERIFIED)

**The template embedder is not optional, even with templates off.** The featurizer always emits four
all-gap template slots and the model runs the embedder on them every cycle; its output is ~29 in magnitude
against |z_init| ~36. Skipping it was my first assumption and was wrong (loop mismatch ~0.3 relative until
it was ported). `mol_mlx/template.py` explains the spec.

**Where one cycle goes at N=97 (MLX, isolated best-of-3, `bench/trunk_profile.py`):** recycle linears 2 ms,
template embedder 20 ms, MSA module 84 ms (21 rows), 48-block stack 697 ms (87%). My earlier guess that the
MSA module and template embedder would be the slow, un-fused parts was wrong; the stack dominates.

**Why the first hybrid was no faster than torch below N~150: not the loop, not the prelude (0.01 s), not the
numpy hand-off (5 ms).** MLX itself runs slower inside a process where torch has done CPU-side work:

| N=97 stack, same inputs, hard-synced | ms |
| --- | --- |
| clean MLX process | 568 |
| + `torch.load` of the checkpoint on CPU | 676 (cpu/wall 0.19 -> 0.35) |
| checkpoint freed again | 590 |
| full torch model + features built in-process | ~760 |
| separate MLX process, another process holding the full torch model on MPS | **562** |

Ruled out, each by measurement: torch's MPS context and resident MPS memory (1.5 / 4 GiB tensors), plain
process memory growth (6 GiB numpy), Python GC (zero collections during the calls; `gc.freeze`/`disable` no
effect), torch thread count, MLX `MAX_OPS_PER_BUFFER`, a fresh Python thread, a dedicated MLX stream,
`mx.clear_cache()`, the memory guard's MPS poll, thermal throttling (12 sustained calls flat at 565 ms),
value dependence (real vs random activations: same time). **The cause is unidentified**; the process
boundary is the cure, so the trunk runs in a persistent worker (`mol_mlx/worker.py`): length-prefixed
pickle over stdin/stdout, arrays through shared memory, the parent draws every cycle's MSA sample up front
(same torch RNG stream; nothing else consumes it in the loop). It starts on first use and lives until the
parent exits; a dead worker raises `WorkerError` (never a silent torch fallback: that could be the memory
blow-up the worker's guard just prevented) and the next request restarts it.

Correction to something I said mid-session: the in-process penalty does NOT grow with N. Direct A/B,
in-process time / worker time: **1.47x at N=97, 1.22x at 128, 1.05x at 192, 1.10x at 256.**

**Trunk benchmark** (`bench/trunk_bench.py 12`: 4 cycles, 64-row MSA, prelude and conversions included,
torch called as `mol pred` does it, hard cap 12 GiB per process, one process per N):

| N | torch | MLX in-process | MLX worker | worker speedup |
| --- | --- | --- | --- | --- |
| 97 | 3.96 s | 3.98 | 2.71 | 1.46x |
| 128 | 6.39 | 6.87 | 5.63 | 1.13x |
| 192 | 24.21 | 16.65 | 15.83 | 1.53x |
| 256 | 42.41 | 29.44 | 26.76 | 1.59x |

**Larger N** (`bench/trunk_bench.py 24 320,384 torch,worker 14`, then worker-only; same 4 cycles and 64-row
MSA; parent cap 14-24 GiB, worker cap 14-22 GiB, no kill):

| N | torch | MLX worker | speedup | torch peak | parent peak | worker MLX peak |
| --- | --- | --- | --- | --- | --- | --- |
| 320 | 61.84 s | 30.42 s | **2.03x** | 14.8 GiB | 14.8 | not recorded |
| 384 | 97.22 s | 46.03 s | **2.11x** | 20.7 GiB | 20.7 | not recorded |
| 512 | not run | 91.86 s | n/a | not run | 7.7 | not recorded |
| 600 | not run | 133.92 s | n/a | not run | 8.3 | **14.2 GiB** |

The worker's MLX peak is the guard's sampled active+cache maximum (a proxy, 0.1 s polling); `ps` RSS cannot
see Metal buffers (it read 3.3 GiB at N=512), so ignore any RSS figure for the worker. Torch was NOT run at
512 or 600: its peak is 14.8 -> 20.7 GiB at 320 -> 384, so roughly 30+ GiB at 512 and more at 600, which is
unsafe on this 48 GB machine. **Extrapolated only, not measured:** torch's time grows ~N^2.0-2.5 over
256-384, which puts it at roughly 175-200 s at N=512 and 235-295 s at N=600, i.e. **~1.9-2.2x** for the
worker. MLX's own scaling is steady (46 -> 92 -> 134 s, exponent ~2.4 throughout). At Mol-1's default 20
cycles the trunk alone is roughly 5x the 4-cycle figure: ~7.7 min at N=512 and ~11 min at N=600 (scaled from
4 cycles; the sampler's cycle with a 1-row MSA is cheaper, so slightly less). Trunk only: diffusion and the
confidence head were not run at these sizes.

**Full prediction at N=600** (`scripts/guarded_pred.py 30 20 pred ...`, `bench/make_job.py 600`; 2 chains, synthetic
64-row target MSA, 1 sample / 4 cycles / 20 steps, `Mol1Oracle`'s `LARGE_INPUT_FLAGS`, `--need_atom_confidence
true`, `-d` left at its default (bf16 label, inert off CUDA: everything runs fp32); parent watchdog cap 30 GiB, worker cap 20 GiB): **completed, exit 0**, CIF +
confidence + full_data written, no fallback to the torch trunk.

| | |
| --- | --- |
| prediction ("Job completed") | **169.3 s** |
| of which MLX trunk (4 cycles, first request incl. kernel warm-up) | 140.1 s |
| everything else (diffusion 20 steps, confidence head, dump) | ~29 s |
| whole process incl. model load and featurization | 219 s |
| torch side, sampled peak (MPS allocator) | **28.2 GiB** (cap 30) |
| MLX worker, sampled MLX peak | 14.2 GiB |

Reference, NOT a same-input baseline: the oracle's table (`mol1_oracle.py`, measured earlier on this machine, 600
tokens, 1 sample / 4 cycles / 20 steps, same LARGE_INPUT_FLAGS, pure torch) records **298.2 s and 33,686 MiB
(32.9 GiB)**. Against that, the hybrid is ~1.76x faster (169 vs 298 s) and its torch-side peak is ~14% lower
(28.2 vs 32.9 GiB). Different session and different input, so read it as indicative: subtracting this run's
~29 s of non-trunk time leaves ~269 s for torch's trunk, inside the 235-295 s I extrapolated from N=256-384.
**Memory:** the torch side still peaks at 28.2 GiB, i.e. diffusion and the confidence head, not the trunk,
set the peak at 600 tokens, and that is 1.8 GiB under the cap used, so this is close to the limit for this
machine. The worker's 14.2 GiB is during the trunk (parent ~8 GiB then); the worker idles at its weights after
`mx.clear_cache()`. The two peaks were not measured as a simultaneous total. This run says nothing about
structure quality (synthetic target, synthetic MSA: pLDDT 64, ipTM 0.14, no clash); it tests speed, memory and
completion. At the default 20 cycles / 200 steps / 5 samples the trunk is ~5x this run's and diffusion ~10x
per sample: not measured.

**Default settings on a real design** (`scripts/guarded_pred.py 33 24 pred ...`; 20 cycles / 200 steps / 5 samples; the
oracle's `LARGE_INPUT_FLAGS`; fp32 throughout: `-d bf16` is inert off CUDA). Input: `mythos_preview_multi_target_tnfa_rank03`
from the third-party `Downloads/protein_binder_design_data_release` (TNF-a trimer 157x3 + 115-residue binder, **586 tokens**,
4623 atoms; used as data only, not one of the user's own designs). **Completed, exit 0**, five CIFs + confidence + full_data.

| | |
| --- | --- |
| prediction ("Job completed") | **1225.6 s (20.4 min)** |
| MLX trunk, 20 cycles | 663.7 s (54%) |
| diffusion (200 steps x 5 samples) + confidence head x5 + dump | ~562 s (46%), ~112 s per sample |
| whole process (model load, MSA search ~125 s, featurization) | 1400 s |
| torch side, sampled peak (MPS allocator) | 23.5 GiB (cap 33) |
| MLX worker, sampled MLX peak | **19.1 GiB** (cap 24; the 20 GiB cap used earlier would have been 0.9 GiB from a kill) |

The five samples agree tightly: ranking score 0.935-0.936, ipTM 0.93, pTM 0.944, pLDDT 92.7, gPDE 0.447, no clash.
Real alignments are deeper than my synthetic 64-row ones: Mol-1 samples up to 16384 rows per cycle (this run: N_msa
2187), which is why `hybrid.auto_chunks` now sizes the MSA row chunk from a 512 MiB budget (149 rows at N=586; the old
constant 2048 would have built a 7.4 GiB intermediate) and the pre-drawn samples go to the worker as uint8 (exact).
`bench/msa_worst_case.py`: 7000 rows at N=586 -> worker peak 18.3 GiB, 36 s per cycle.

**This run used a remote MSA server (unintended, user later said fine for now).** The binder chain came in the data
release's legacy inline format (`unpairedMsa` string, no path), so Mol-1's `need_msa_search` fired and searched BOTH
chains on `https://api.colabfold.com` (~125 s), replacing the supplied target alignment with the server's. Sequences sent:
the TNF-a target and the 115-residue binder. Any chain without `unpairedMsaPath`/`pairedMsaPath` triggers this in plain
`mol pred` too; `Mol1Oracle` supplies explicit MSAs for every chain to avoid it. `mol1-mlx` does not block it.

Read these with care. **Torch's baseline is noisy**: N=192 was 18.94 s in an earlier identical run (1.19x
then) and 24.21 s here; N=256 was 39.29 s earlier (1.32x in-process then). The MLX columns repeat to ~1%
(N=192: 15.87 vs 15.83; N=256: 29.75 vs 29.44). Quote the speedups as ranges: ~1.1-1.5x at N=97-128,
~1.2-1.5x at 192, ~1.5-1.6x at 256. N=128 looks low against its neighbours and was not re-run. Peak held
memory was 6.0-9.4 GiB at these sizes.

**End to end, the real CLI** (N=97, 4 cycles, 20 steps, 1 sample, `-d` at its default: inert off CUDA, so fp32 throughout; per-seed time in
one process, 3 seeds): torch 7.85 / 4.53 / 4.53 s, `mol1-mlx` 7.76 / 3.57 / 3.58 s. **The first
prediction in a process costs the same either way (~7.8 s): worker start-up and kernel warm-up cancel the
gain, so the saving appears from the second prediction on.** That suits a design loop (thousands of
predictions per process) and not a one-off. Default is 20 cycles, so the trunk's share, and the gain, is
larger than in these 4-cycle runs; not measured end to end at 20 cycles above N=97.

## Fusion of the elementwise passes (item 3: DONE, small)

Probed on the real block-8 weights at N=256 (`bench/fusion_probe.py`, `bench/trimul_fusion.py`), each variant
checked against the current output before its time was read:

| op | before | after | what changed |
| --- | --- | --- | --- |
| Transition | 22.0 ms | 18.2 | one concatenated a/b matmul + one compiled `silu(a)*b` pass |
| Triangle multiplication | 15.2 ms | 13.9 | compiled `sigmoid(g)*p` gates (a, b, and the output gate) |

Both are in `mol_mlx/layers.py` (`mx.compile(shapeless=True)`, so a new N does not retrace). Whole 48-block
stack at N=256, interleaved ABABAB in one process (`bench/fusion_ab.py`): unfused 4.85 s median, fused 4.56 s,
**1.065x**, output differing 6.3e-6 relative. Things that did NOT help: a 5-way concatenated tri-mul
projection (15.3 -> 23.5 ms, slower: the [N,N,5c] intermediate and strided slices cost more than the shared
read saves); chunking the transition to keep intermediates cache-resident (8-64 row chunks within 1% of
unchunked; the old separate-matmul transition got slower, 22 -> 24-27 ms). Tri-mul is now ~13.9 ms against a
~12 ms floor (5 projections 6.3 + contraction 2.2 + two layer norms 2.4 + output linear), so little is left.

## Diffusion: the cache setting, not the compute, was most of the time (MEASURED; changes the port's case)

I recommended porting diffusion because the default-settings run spent 562 of 1226 s (46%) in diffusion + confidence.
Measuring a real denoising step first (`bench/diffusion_cache_probe.py`, real Mol-1 on MPS, trunk in the MLX worker,
same model/features/seed, modes interleaved, `Mol1Oracle`'s other LARGE_INPUT_FLAGS applied) found that most of that
time is repeated work caused by `diffusion_cache_mode=off`, which **`Mol1Oracle` sets above 400 tokens and which I
added** (Mol-1's own "Large inputs" recipe keeps the default `full` and relies on the row chunking that only applies
when the cache is on). With the cache off, every one of the 200 steps x 5 samples recomputes the N x N pair
conditioning (896->448 projection + two 448-wide transitions), every block's pair-bias projection, and the atom-pair
features. The oracle's flags were validated at 20 steps x 1 sample, where this costs ~10 s and is invisible.

| N | cache full | cache off | off costs |
| --- | --- | --- | --- |
| 200 | ~137 ms/step | ~190 ms/step | 1.4x (N=200 run had no worker and no chunk flags) |
| 384 | ~91 ms/step | ~250-265 ms/step | 2.7x |
| 586 | ~145-151 ms/step | ~513-554 ms/step | **3.4-3.7x** |

Per-step values include the one-time cache preparation amortised over only 20 steps, so the cache-on figures
overstate the steady state. **Memory: no benefit from `off`.** Sampled torch MPS peak at N=586, one mode per
process: 27.3 GiB (full) vs 27.0 GiB (off); the cache is ~1.1 GB resident and the peak is set elsewhere. (The
sampled peak covers the whole process, so this bounds the cache-on increase at ~0.3 GiB.) The ~0.5 s/step the
default-settings run showed is exactly the cache-off cost. Estimated effect of `full` on that run (NOT re-run end to
end): diffusion ~500 s -> ~130-150 s, run ~1226 s -> ~860-880 s (-28 to -30%).

**Scope:** only `Mol1Oracle` above 400 tokens. Protenix's `--enable_cache` defaults to true and `ProtenixOracle`
does not override it, so the ~160 s Protenix oracle calls in the search runs were already cache-on.

Component split of a cache-on denoiser call at N=200 (synced, ms/step): token transformer 38.7 (49%), atom attention
encoder 22.9 (29%), atom decoder 13.6 (17%), s-conditioning 1.4; 78.5 total. The confidence head is negligible
(~0.1 s at N=200, ~2 s at N=586). So a diffusion port would target the 24-block transformer and the atom attention;
the confidence head is not worth porting.

**What this does to the port's case:** with the cache on, diffusion at N=586 is ~130-150 s of an ~870 s run (~16%);
an MLX diffusion port at 2x would save ~70 s (~8%). Weak. The case rests on the Protenix oracle (cache already on),
where diffusion is plausibly 50-70% of a ~160 s call at N=287 (estimated from the cache-on per-step cost above, NOT
measured on Protenix). Profile one Protenix call (vary `-p`, `-c`) before investing.

## NOT DONE

The input embedder (atom attention encoder), the diffusion module (206M params, 200 steps x samples) and
the confidence head (38M) still run in torch/MPS; so does the sampler. The trunk's share of a default run
was derived from three earlier MPS runs at 245 tokens (241.65 s at 20 cycles/200 steps/5 samples; 184.19 s
at 2 samples; 39.74 s at 4 cycles/50 steps/2 samples): diffusion ~19.2 s per sample at 200 steps and
everything else ~7.3-7.5 s per cycle, so the trunk is ~60% of a default run at 5 samples and ~79% at 2
(derived at N=245 only; it grows with N). Not done: a same-input torch baseline at N>=512 (torch's full
prediction peaks ~33 GiB at 600 on this machine); default settings (20 cycles / 200 steps / 5 samples) end to
end above N=97; structure quality at any size (all test inputs are synthetic), MC-dropout / batched / constraint features
(the hybrid falls back to torch for those), and `MOL_BIN` wiring for the oracle (left to the user;
`.env.local` was not touched).

## Recommended order

1. **Default-settings run at 600 tokens** (20 cycles, 200 steps, 5 samples), and a real design as input: the
   N=600 run above used 4 cycles / 20 steps / 1 sample on a synthetic input and peaked at 28.2 GiB of a 30 GiB
   cap. Ask first, and keep the watchdog and `LARGE_INPUT_FLAGS`.
2. **Cycles knob, no port needed.** `Mol1Oracle` (RFD4-Proteina) does not pass `-c`, so it runs Mol-1's
   default of 20 recycles. Mol-1's report states 10 cycles costs 0.01-0.03 AUC points, **on its natural
   antibody-antigen benchmarks**, not on separating designs from non-binders, so this is the user's call.
3. **Point `MOL_BIN` at `bin/mol1-mlx`** (user's `.env.local`; the oracle's argv is unchanged). The oracle's
   RFD4 preflight also fails today because CUDA is required: a separate blocker (`HANDOFF_MOL1_ORACLE.md`).
4. Identify the in-process slowdown's cause. It would remove the worker (and its first-call cost) if it is
   fixable, and it may affect other MLX-plus-torch code on this machine.
5. Skip exactly-dead `attention_pair_bias` (finding 2); measure the gain first.
6. Port diffusion + confidence head. MEASURED at N=586 / 5 samples / 20 cycles: they are ~46% of the prediction now
   (562 of 1226 s), the largest remaining share. Profile one denoiser step and one confidence-head call first (the log
   gives only totals), then confidence head (4 Pairformer blocks, reuses ours), DiT (24 blocks, c_token 768), atom
   encoder/decoder (3+3 blocks, windowed attention over ~4600 atoms) last. See the notes in the memory file on the
   process boundary and RNG.

Kernel-level work is now low priority: the attention kernel is ~7 ms of a ~97 ms Pairformer block, and
v2's 3.68 ms at N=256 is close to v1's measured no-math floor (3.4 ms), a hint (not proof) that little
is left in it. Strided q/k/v loads and an overlapping tail tile would each be a few percent at most.

## Rules this session ran under (the user's)

- **Memory:** a 500-token full-model profile was killed by memory pressure and the user objected.
  Since then every test is small (48/72/97 tokens for correctness), isolated ops for
  benchmarks, a hard watchdog cap (`mol_mlx/guard.py`, kills the process; `mx.set_memory_limit`
  alone is advisory), and no full-model run above ~128 tokens without asking.
- GPU runs are allowed (the earlier "no GPU until I say" was lifted).

## Reproducing

```bash
conda activate mol1-mps   # has mlx 0.32.2 + torch + mol (editable ~/Mol-1)
cd ~/Mol-1-MLX
PYTHONPATH=. python scripts/convert_weights.py            # -> weights/ (2.1 GB)
PYTHONPATH=. python scripts/capture_refs.py 48 72 97      # -> refs/ (1.7 GB), torch CPU, 12 GiB cap
PYTHONPATH=. python scripts/capture_recycle_refs.py 48 97 # recycle refs (3 cycles, torch CPU)
PYTHONPATH=. pytest tests -q                              # 134 pass
PYTHONPATH=. python bench/stack_bench.py 10               # stack vs torch/MPS, 10 GiB cap
PYTHONPATH=. python bench/tri_attn_capacity.py 6          # kernel vs dense vs fused, 6 GiB cap
PYTHONPATH=. python bench/trunk_bench.py 12               # torch vs MLX in-process vs worker, N=97-256
PYTHONPATH=. python bench/fusion_ab.py                    # fused vs unfused stack, interleaved
PYTHONPATH=. python bench/stack_in_torch_process.py       # the in-process slowdown, N=97
bin/mol1-mlx pred -i job.json -o out --device mps         # the CLI; same flags as `mol pred`
```
