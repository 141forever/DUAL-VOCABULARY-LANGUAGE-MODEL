# DVLM

Code for learning a teacher-to-student vocabulary projection and using it for on-policy knowledge distillation between language models with different tokenizers.

This research release contains two stages:

1. **Projection training:** freeze the teacher and train a linear output head that predicts tokens in the student's vocabulary from aligned teacher hidden states.
2. **On-policy distillation (OPD):** generate student completions, obtain aligned teacher predictions through the learned head, and update the student with the custom GOLD trainer.

The example configuration uses a Qwen3-4B teacher and a Llama-3.2-1B-family student. Projection training loads only the student tokenizer; OPD loads both models.

> **Release status:** this is a research code snapshot. Projection training exposes a command-line interface. OPD requires local model/data paths and integration of the supplied trainer into a compatible TRL source tree. Dependency versions, a complete environment specification, and benchmark evaluation scripts are not included.

## Repository contents

```text
.
├── DVLM training.py          # Token alignment, preprocessing, and projection-head training
├── DVLM-based OPD.py         # Model/data setup and on-policy distillation entry point
├── gold_rewrite_for_DVLM.py  # Custom GOLDTrainer with DVLM teacher inference and loss
├── data_8k.parquet     # Bundled data file; inspect its schema before use
└── tmp                      # Auxiliary file
```

## Method overview

The projection stage aligns student and teacher tokenizations of the same text. It builds a supersequence with custom attention masks and position IDs to extract teacher hidden states at the required student-token boundaries. A trainable linear head maps these states to the student vocabulary and is optimized with next-token cross-entropy. The teacher remains frozen.

The projection is initialized from the teacher's output embeddings: directly corresponding tokens reuse a teacher row; other tokens use the mean of the rows for their teacher-token decomposition, with a fallback when no valid decomposition is available.

During OPD, the learned projection replaces the teacher's language-model head. The custom trainer aligns teacher predictions with student-generated completions and applies its hybrid distillation objective, combining matched-token distribution matching with a sorted-probability loss for unmatched tokens.

## Environment

Use a Linux environment with an NVIDIA GPU and a CUDA-enabled PyTorch installation. Projection training initializes an NCCL process group, so it must be launched with `torchrun`, including for a single GPU.

The code imports the following packages:

| Component | Dependencies |
| --- | --- |
| Projection training | `torch`, `transformers`, `datasets`, `tqdm`, `wandb` |
| OPD | The above, plus compatible `trl`, `accelerate`, `peft`, and `vllm` installations |
| Parquet inspection | `pyarrow` |

After installing a CUDA-compatible PyTorch build, a starting point for the projection-stage dependencies is:

```bash
python -m pip install transformers datasets pyarrow tqdm wandb
```

These dependencies are not version-pinned in this release. OPD additionally relies on experimental/internal TRL APIs and version-sensitive vLLM imports; installing an arbitrary current TRL release is not sufficient to establish compatibility.

Download the repository files and run the following commands from their containing directory. Filenames containing spaces must be quoted.

## Stage 1: Train the projection head

### Prepare a text corpus

The implemented loader reads a **local Parquet file**, with one text column selected by `--text_col` (default: `text`). Each value must be a string. Rows with at most 20 non-whitespace characters are excluded before preprocessing.

Inspect the bundled file before deciding how to use it:

```bash
python -c "import pyarrow.parquet as pq; print(pq.read_schema('data_8k.parquet'))"
```

If it contains the required text column, it can be supplied as `--dataset_name`; otherwise, prepare a Parquet file with that column. The filename alone does not establish its schema or suitability for either stage.

### Build the cache and train

The following example uses the original student vocabulary by passing an empty `--reasoning_tokens` list. Replace the model locations and corpus path as needed.

```bash
mkdir -p cache

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  "DVLM training.py" \
  --student_model meta-llama/Llama-3.2-1B-Instruct \
  --teacher_model Qwen/Qwen3-4B \
  --dataset_name /path/to/corpus.parquet \
  --text_col text \
  --cache_path cache/aligned.pkl \
  --if_write_cache \
  --max_tokens 512 \
  --max_seq_len 2048 \
  --batch_size 4 \
  --teacher_bsz 32 \
  --lr 1e-5 \
  --epochs 1 \
  --grad_accum 1 \
  --dtype bfloat16 \
  --output_dir output_projection \
  --reasoning_tokens
```

`--if_write_cache` builds the alignment cache **and then runs training**. To reuse an existing cache, omit that flag and keep the same `--cache_path` and tokenizer configuration. Create the cache's parent directory beforehand.

### Important options

| Argument | CLI default | Meaning |
| --- | --- | --- |
| `--student_model` | `meta-llama/Llama-3.2-1B` | Student tokenizer source |
| `--teacher_model` | `Qwen/Qwen3-4B` | Frozen teacher model and tokenizer |
| `--dataset_name` | Placeholder path | Local input Parquet file |
| `--text_col` | `text` | Text column in the input file |
| `--cache_path` | None | Required alignment-cache path |
| `--max_samples` | `-1` | Maximum source rows considered; `-1` uses all rows |
| `--max_tokens` | `512` | Skip examples exceeding this length under either tokenizer when building the cache |
| `--max_seq_len` | `2048` | Supersequence length setting |
| `--batch_size` | `4` | Per-GPU batch size |
| `--teacher_bsz` | `32` | Teacher forward batch-size setting |
| `--lr` | `1e-5` | Projection learning rate |
| `--epochs` | `1` | Training epochs |
| `--grad_accum` | `1` | Gradient accumulation steps |
| `--dtype` | `bfloat16` | Teacher precision; also supports `float16` |
| `--init_head_path` | None | Initialize from an existing projection with an exactly matching weight shape |
| `--reasoning_tokens` | `<think> </think> <answer> </answer>` | Additional student tokens; pass the flag without values to disable |

`--max_tokens` filters long examples rather than truncating them. Rebuild the cache when changing tokenizers, added tokens, or preprocessing settings. The current loader does not use `--dataset_subset`.

### Outputs

```text
output_projection/
├── proj_final.pt
├── proj_epoch<N>.pt
├── proj_step<N>.pt                         # At configured save intervals
├── reasoning_token_ids.json
└── student_tokenizer_with_reasoning_tokens/
```

Projection checkpoints contain a `weight` tensor with shape `[student_vocabulary_size, teacher_hidden_size]`. They contain the projection head, not a complete model or optimizer/scheduler state. `--init_head_path` initializes weights; it does not resume a full training run.

## Stage 2: On-policy distillation

### Integrate the custom trainer

`gold_rewrite_for_DVLM.py` contains package-relative imports and is intended to be integrated into a compatible TRL GOLD package. It is not a standalone executable.

1. Prepare a compatible TRL source checkout containing `experimental/gold`, its `gold_config.py`, and the utilities imported by the supplied trainer.
2. Integrate the supplied trainer as that package's `gold_trainer.py`, retaining the package structure and its `GOLDTrainer` export. Keep a backup of the original implementation.
3. Verify that `GOLDConfig` accepts the extended ULD settings used in `DVLM-based OPD.py`, including `use_extended_uld` and `uld_use_hybrid_loss`.
4. Adjust the entry-point import to the installed package layout. The snapshot uses `trl.trl.experimental.gold`; an installed checkout may instead expose `trl.experimental.gold`.

The exact TRL revision and a matching configuration implementation are not supplied, so this integration must be checked in the intended training environment.

### Configure models, projection, and data

Edit `DVLM-based OPD.py` before launching:

| Setting | Required configuration |
| --- | --- |
| `student_name` | Local student model directory |
| `teacher_name` | Local teacher model directory matching Stage 1 |
| `new_head_path` | Stage 1 checkpoint, e.g. `output_projection/proj_final.pt` |
| Training `Dataset.from_parquet(...)` | Replace `xxx.parquet` with the training data path |
| Test `Dataset.from_parquet(...)` | Replace `xxx.parquet` with the test data path |
| `changed_teacher_tokenizer` | Tokenizer describing the projected output vocabulary, i.e. the student tokenizer used in Stage 1 |
| `GOLDConfig(...)` | Batch sizes, precision, output location, and generation-server connection |

Model and tokenizer loading in this entry point uses `local_files_only=True`; prepare local model files first.

**Keep the two teacher vocabularies distinct.** The teacher still consumes tokens from its original tokenizer, while its replaced head outputs scores in the Stage 1 student vocabulary. Accordingly, keep the original teacher tokenizer for inputs and set `changed_teacher_tokenizer` to the projection's output tokenizer. The hard-coded `Qwen3-4B` value must be changed for a projection trained on a Llama vocabulary.

If Stage 1 added reasoning tokens, load the saved augmented tokenizer for the student, resize its embeddings/output head as needed, and ensure the generation server uses the same vocabulary. The provided OPD entry point does not perform that resizing automatically. The Stage 1 example above disables token addition to avoid introducing this extra configuration step.

### Dataset format

The OPD collator expects training examples with a `messages` field containing chat messages, for example:

```json
{
  "messages": [
    {"role": "user", "content": "Solve the following problem: ..."}
  ]
}
```

The collator removes trailing assistant messages and generates a fresh student completion. Each example must retain a valid prompt after that removal.

The current test-data loader expects `prompt`, `target`, and `nums`, converting `prompt` into `messages`. It randomly selects 1,000 examples, so provide at least 1,000 test rows or change that sampling step. Test data is loaded even though `do_eval=False` is set in the training configuration.

### Generation service and launch

The supplied OPD configuration uses a **TRL-compatible vLLM generation server** at `127.0.0.1:8000`. Start the server using the command supported by your compatible TRL/vLLM environment, with the student model and tokenizer prepared above. The custom trainer uses TRL's `VLLMClient`; a generic OpenAI-compatible inference endpoint is not a substitute for that server interface.

After completing the integration and path configuration:

```bash
python "DVLM-based OPD.py"
```

The example settings include a learning rate of `2e-6`, two epochs, BF16, a per-device training batch size of `21`, and a maximum completion length of `1024`. Adjust batch sizes and GPU allocation for the memory required by the student, teacher, and generation service.

## Implementation notes

- **Use one GPU for projection training initially.** The source explicitly identifies unresolved multi-rank gradient-normalization and synchronization issues.
- **The projection step counter starts at `10200`.** Change it to `0` for a fresh run if you want checkpoint names and logs to start from the beginning; the existing value does not restore training state.
- **Check tokenizer API compatibility.** The OPD collator indexes the result of `apply_chat_template` as a dictionary. Match its return format to the installed Transformers version, for example by requesting `return_dict=True` where supported.
- **Evaluation is not packaged as a reproducible benchmark workflow.** The snapshot contains trainer evaluation code but no standalone benchmark commands or reported result tables.

## Citation and license

Paper metadata and a repository-level license are not included in this snapshot. Add the verified paper citation and intended license when they are available for release.
