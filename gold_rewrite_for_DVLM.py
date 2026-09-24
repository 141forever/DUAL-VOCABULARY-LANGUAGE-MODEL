# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pdb
import random
import textwrap
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, Optional, cast, List, Tuple, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction
from transformers.utils.import_utils import (
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)
from ...data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from ...extras.profiling import profiling_decorator
from ...extras.vllm_client import VLLMClient
from ...import_utils import is_vllm_available
from ...models import prepare_deepspeed
from ...models.utils import unwrap_model_for_generation
from ...trainer.sft_trainer import SFTTrainer
from ...trainer.utils import (
    create_model_from_path,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
    pad,
)
from ..utils import DataCollatorForChatML
from .gold_config import GOLDConfig

if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

def print_prompt_completions_sample_uld(
    prompts: list[str],
    completions: list[str],
    step: int,
    num_samples: int = None,
) -> None:
    """
    Print out a sample of model completions to the console with multiple reward metrics.

    This function creates a nicely formatted table showing prompt-completion pairs, useful for monitoring model outputs
    during training. It requires the `rich` library to be installed.

    Args:
        prompts (`list[str]`):
            List of prompts.
        completions (`list[str]`):
            List of completions corresponding to the prompts.
        rewards (`dict[str, list[float]]`):
            Dictionary where keys are reward names and values are lists of rewards.
        advantages (`list[float]`):
            List of advantages corresponding to the prompts and completions.
        step (`int`):
            Current training step number, used in the output title.
        num_samples (`int` or `None`, *optional*, defaults to `None`):
            Number of random samples to display. If `None` (default), all items will be displayed.

    Example:
    ```python
    >>> from trl.trainer.utils import print_prompt_completions_sample

    >>> prompts = ["The sky is", "The sun is"]
    >>> completions = [" blue.", " in the sky."]
    >>> rewards = {"Correctness": [0.123, 0.456], "Format": [0.789, 0.101]}
    >>> advantages = [0.987, 0.654]
    >>> print_prompt_completions_sample(prompts, completions, rewards, advantages, 42)
    ╭──────────────────────────── Step 42 ─────────────────────────────╮
    │ ┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┓ │
    │ ┃ Prompt     ┃ Completion   ┃ Correctness ┃ Format ┃ Advantage ┃ │
    │ ┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━┩ │
    │ │ The sky is │  blue.       │        0.12 │   0.79 │      0.99 │ │
    │ ├────────────┼──────────────┼─────────────┼────────┼───────────┤ │
    │ │ The sun is │  in the sky. │        0.46 │   0.10 │      0.65 │ │
    │ └────────────┴──────────────┴─────────────┴────────┴───────────┘ │
    ╰──────────────────────────────────────────────────────────────────╯
    ```
    """
    if not is_rich_available():
        raise ImportError(
            "The function `print_prompt_completions_sample` requires the `rich` library. Please install it with "
            "`pip install rich`."
        )
    console = Console()
    table = Table(show_header=True, header_style="bold white", expand=True)

    # Add columns
    table.add_column("Prompt", style="bright_yellow")
    table.add_column("Completion", style="bright_green")

    # Some basic input validation
    if num_samples is not None:
        if num_samples >= len(prompts):
            num_samples = None
        elif num_samples <= 0:
            return

    # Subsample data if num_samples is specified
    if num_samples is not None:
        indices = random.sample(range(len(prompts)), num_samples)
        prompts = [prompts[i] for i in indices]
        completions = [completions[i] for i in indices]

    for i in range(len(prompts)):
        table.add_row(Text(prompts[i]), Text(completions[i]))
        table.add_section()  # Adds a separator between rows

    panel = Panel(table, expand=False, title=f"Step {step}", border_style="bold white")
    console.print(panel)


def build_teacher_inputs_from_texts(
    tokenizer: PreTrainedTokenizerBase,
    prompt_texts: list[str],
    completion_texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Tokenize teacher prompts/completions and produce tensors ready for GOLD loss."""

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    prompt_token_ids = tokenizer(prompt_texts, add_special_tokens=True)["input_ids"]
    completion_token_ids = tokenizer(completion_texts, add_special_tokens=False)["input_ids"]

    sequences: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    prompt_lengths: list[int] = []

    for prompt_ids, completion_ids in zip(prompt_token_ids, completion_token_ids, strict=True):
        # Remove trailing EOS from prompt so completions can extend cleanly
        if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
            prompt_ids = prompt_ids[:-1]

        prompt_lengths.append(len(prompt_ids))
        sequence = list(prompt_ids)
        sequence.extend(completion_ids)
        if eos_token_id is not None:
            sequence.append(eos_token_id)

        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        sequences.append(seq_tensor)
        attention_masks.append(torch.ones_like(seq_tensor))

        labels = seq_tensor.clone()
        labels[: len(prompt_ids)] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        labels_list.append(labels)

    teacher_input_ids = pad(
        sequences,
        padding_side="right",
        padding_value=pad_token_id if pad_token_id is not None else 0,
    )
    teacher_attention_mask = pad(attention_masks, padding_side="right", padding_value=0).bool()
    teacher_labels = pad(labels_list, padding_side="right", padding_value=-100)

    if eos_token_id is not None:
        for row in range(teacher_attention_mask.size(0)):
            valid = (
                teacher_input_ids[row] != pad_token_id
                if pad_token_id is not None
                else teacher_attention_mask[row].bool()
            )
            if valid.any():
                last_idx = valid.nonzero(as_tuple=True)[0][-1]
                teacher_attention_mask[row, last_idx + 1 :] = False

    teacher_prompt_length = max(prompt_lengths) if prompt_lengths else 0

    return teacher_input_ids, teacher_labels, teacher_attention_mask, teacher_prompt_length


class ULDLoss(nn.Module):
    """
    Universal Logit Distillation Loss.
    """

    def __init__(self, config: GOLDConfig, student_tokenizer=None, teacher_tokenizer=None,changed_teacher_tokenizer=None):
        super().__init__()
        self.crossentropy_weight = config.uld_crossentropy_weight
        self.distillation_weight = config.uld_distillation_weight
        self.student_temperature = config.uld_student_temperature
        self.teacher_temperature = config.uld_teacher_temperature
        self.skip_student_eos = config.uld_skip_student_eos
        self.skip_teacher_eos = config.uld_skip_teacher_eos
        self.use_extended_uld = config.use_extended_uld
        self.ignore_index = -100

        # Add tokenizers for enhanced alignment
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.changed_teacher_tokenizer = changed_teacher_tokenizer 

        # Hybrid ULD configuration
        self.use_hybrid_loss = getattr(config, "uld_use_hybrid_loss", False)
        self.hybrid_matched_weight = getattr(config, "uld_hybrid_matched_weight", None)
        self.hybrid_unmatched_weight = getattr(config, "uld_hybrid_unmatched_weight", None)
        self.beta = getattr(config, "beta", 1.0)  # For JSD loss in hybrid matched tokens

        # Initialize vocabulary mapping for hybrid loss
        self._vocab_mapping = None
        self._teacher_matched_ids = None
        self._student_matched_ids = None
        if self.use_hybrid_loss and student_tokenizer is not None and teacher_tokenizer is not None and self.changed_teacher_tokenizer is not None:
            self._initialize_vocabulary_mapping()

    def __call__(
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
    ):
        """
        Compute ULD loss with GKD trainer interface.

        Args:
            student_logits: Student model logits [batch_size, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
            student_labels: Student target labels [batch_size, seq_len]
            teacher_labels: Teacher target labels [batch_size, seq_len]
            student_input_ids: Student input token IDs [batch_size, seq_len]
            teacher_input_ids: Teacher input token IDs [batch_size, seq_len]

        Returns:
            Total loss (cross-entropy + distillation)
        """
        # Compute cross-entropy loss for student
        if self.crossentropy_weight > 0:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = student_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            crossentropy_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            crossentropy_loss = self.crossentropy_weight * crossentropy_loss
        else:
            crossentropy_loss = 0.0

        # Compute distillation loss using ULD approximation
        distillation_loss = self._compute_distillation_loss(
            student_logits,
            teacher_logits,
            student_labels,
            teacher_labels,
            student_input_ids,
            teacher_input_ids,
        )

        return crossentropy_loss + distillation_loss

    def _initialize_vocabulary_mapping(self):
        """Initialize vocabulary mapping for hybrid ULD loss."""
        # Computing vocabulary mapping for hybrid ULD

        student_vocab = self.student_tokenizer.get_vocab()
        changed_teacher_vocab = self.changed_teacher_tokenizer.get_vocab() 

        # Create reverse mapping for student
        student_token_to_id = dict(student_vocab.items())

        vocab_mapping = {}
        teacher_matched_ids = set()
        student_matched_ids = set()

        for token_str, teacher_id in changed_teacher_vocab.items(): 
            
            if token_str in student_token_to_id:
                student_id = student_token_to_id[token_str]
                vocab_mapping[teacher_id] = student_id
                teacher_matched_ids.add(teacher_id)
                student_matched_ids.add(student_id)

        self._vocab_mapping = vocab_mapping
        self._teacher_matched_ids = teacher_matched_ids
        self._student_matched_ids = student_matched_ids

    def _compute_distillation_loss(
        self,
        student_logits,
        teacher_logits,
        student_labels,
        teacher_labels,
        student_input_ids,
        teacher_input_ids,
    ):
        """
        Compute the Universal Logit Distillation loss with token mapping.

        This version uses actual input_ids for accurate token mapping and multiplies probabilities for split tokens.
        Both student_input_ids and teacher_input_ids are required for optimal alignment.
        """
        # Get answer regions (same as original)
        student_answer_index, student_answer_size = self._get_start_and_size_answers(student_labels)
        teacher_answer_index, teacher_answer_size = self._get_start_and_size_answers(teacher_labels)

        if self.skip_student_eos:
            student_answer_size = [size - 1 for size in student_answer_size]
        if self.skip_teacher_eos:
            teacher_answer_size = [size - 1 for size in teacher_answer_size]

        # Handle edge case where all answer sizes are 0
        if (
            not student_answer_size
            or not teacher_answer_size
            or max(max(student_answer_size), max(teacher_answer_size)) <= 0
        ):
            return (
                torch.zeros(1, device=student_logits.device, requires_grad=True)
                * student_logits.sum()
                * 1e-8
            )

        batch_size = student_logits.size(0)
        distillation_losses = []

        for i in range(batch_size):
            # Get answer regions for this batch item
            student_start = student_answer_index[i]
            student_size = student_answer_size[i]
            teacher_start = teacher_answer_index[i]
            teacher_size = teacher_answer_size[i]

            if student_size <= 0 or teacher_size <= 0:
                loss_i = student_logits[i].sum() * 0.0
                distillation_losses.append(loss_i)
                continue
            # Extract answer logits
            student_answer_logits = student_logits[i, student_start : student_start + student_size]
            teacher_answer_logits = teacher_logits[i, teacher_start : teacher_start + teacher_size]

            # Convert to probabilities
            student_probs = F.softmax(student_answer_logits / self.student_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_answer_logits / self.teacher_temperature, dim=-1)

            # Get token IDs for mapping (always use actual input_ids)
            student_token_ids = student_input_ids[
                i, student_start : student_start + student_size
            ].tolist()
            teacher_token_ids = teacher_input_ids[
                i, teacher_start : teacher_start + teacher_size
            ].tolist()

            if self.use_extended_uld:
                # Build alignment groups directly from token ids using greedy text matching
                student_alignment_groups, teacher_alignment_groups = (
                    self._build_alignment_groups_from_ids(student_token_ids, teacher_token_ids)
                )

                # Merge student probabilities using student alignment groups
                student_aligned = self._merge_probabilities_with_alignment_groups(
                    student_probs, student_alignment_groups
                )

                # Merge teacher probabilities using teacher alignment groups
                teacher_aligned = self._merge_probabilities_with_alignment_groups(
                    teacher_probs, teacher_alignment_groups
                )
            else:
                min_length = min(len(student_token_ids), len(teacher_token_ids))
                student_aligned = student_probs[:min_length, :]
                teacher_aligned = teacher_probs[:min_length, :]

            # Apply ULD loss computation
            if self.use_hybrid_loss and self._vocab_mapping is not None:
                # Use hybrid approach: direct comparison for matched tokens, sorting for unmatched
                aligned_loss = self._compute_hybrid_uld_loss(student_aligned, teacher_aligned)
            else:
                # Original approach: sort all probabilities
                student_sorted = student_aligned.sort(dim=-1, descending=True).values
                teacher_sorted = teacher_aligned.sort(dim=-1, descending=True).values

                # Pad vocabularies to same size
                student_vocab_size = student_sorted.size(-1)
                teacher_vocab_size = teacher_sorted.size(-1)
                max_vocab_size = max(student_vocab_size, teacher_vocab_size)

                if student_vocab_size < max_vocab_size:
                    student_sorted = F.pad(student_sorted, (0, max_vocab_size - student_vocab_size))
                if teacher_vocab_size < max_vocab_size:
                    teacher_sorted = F.pad(teacher_sorted, (0, max_vocab_size - teacher_vocab_size))

                # Compute L1 distance (ULD approach)
                aligned_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
                aligned_loss /= student_aligned.size(0)  # Normalize by sequence length
            distillation_losses.append(aligned_loss)

        distillation_loss = torch.stack(distillation_losses).mean()
        return self.distillation_weight * distillation_loss

    def _build_alignment_groups_from_ids(self, student_token_ids, teacher_token_ids):
        """
        Build alignment groups using a greedy substring-equality algorithm on decoded token pieces.

        Args:
            student_token_ids: List[int]
            teacher_token_ids: List[int]

        Returns:
            Tuple[List[List[int]], List[List[int]]]: student and teacher alignment groups
        """

        def to_canonical_pieces(tok, ids):
            pieces = []
            prev = ""
            for k in range(len(ids)):
                # IMPORTANT: Do NOT skip special tokens - we need to align them too
                cur = tok.decode(
                    ids[: k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                # Extract the incremental addition (may include spaces/ZWJ/etc.)
                pieces.append(cur[len(prev) :])
                prev = cur
            return pieces

        s_pieces = to_canonical_pieces(self.student_tokenizer, student_token_ids)
        t_pieces = to_canonical_pieces(self.teacher_tokenizer, teacher_token_ids)

        i = j = 0
        s_buf = t_buf = ""
        s_group = []
        t_group = []
        s_groups = []
        t_groups = []

        def flush():
            if s_group and t_group:
                s_groups.append(s_group.copy())
                t_groups.append(t_group.copy())

        # Greedily accumulate pieces until substrings match, then flush
        while i < len(s_pieces) or j < len(t_pieces):
            if s_buf == t_buf and s_buf != "":
                flush()
                s_buf = t_buf = ""
                s_group = []
                t_group = []
                continue

            if s_buf == "" and i < len(s_pieces):
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
                continue
            if t_buf == "" and j < len(t_pieces):
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
                continue

            if len(s_buf) <= len(t_buf):
                if i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                elif j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
            else:
                if j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                elif i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1

        # Flush any remainder if both sides accumulated something
        if s_buf == t_buf and s_group and t_group:
            flush()
        elif s_group or t_group:
            # Handle remaining unmatched tokens by forcing a flush
            # This ensures both sides have the same number of alignment groups
            if s_group or t_group:
                # Ensure both groups have content (even if empty list)
                if not s_group:
                    s_group = []
                if not t_group:
                    t_group = []
                # Force flush even if buffers don't match
                if s_group or t_group:
                    s_groups.append(s_group.copy() if s_group else [])
                    t_groups.append(t_group.copy() if t_group else [])

        return s_groups, t_groups

    def _merge_probabilities_with_alignment_groups(self, probs, alignment_groups):
        """
        Merge probabilities based on alignment groups.

        Args:
            probs: Probability tensor [seq_len, vocab_size]
            alignment_groups: List of alignment groups (each group is a list of positions to merge)

        Returns:
            Merged probability tensor [num_groups, vocab_size]
        """
        if not alignment_groups:
            return probs

        # Create aligned tensor
        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device)

        # Process each alignment group
        for group_idx, group in enumerate(alignment_groups):
            # Handle probability merging
            if len(group) > 1:
                # Multiple tokens map to this group - merge them
                eps = 1e-8
                logp = torch.log(probs[group[0]].clamp_min(eps))
                for idx in group[1:]:
                    if idx < probs.size(0):
                        logp = logp + torch.log(probs[idx].clamp_min(eps))
                aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
            elif len(group) == 1:
                aligned_probs[group_idx] = probs[group[0]]
            else:
                # No tokens map to this group
                aligned_probs[group_idx] = torch.zeros_like(probs[0])

        return aligned_probs

    def _compute_hybrid_uld_loss(self, student_aligned, teacher_aligned):
        """
        Compute hybrid ULD loss on aligned probability distributions. This method:
        1. Directly compares probabilities for tokens with matching vocabulary entries
        2. Uses sorting approach only for tokens with different vocabulary entries

        Args:
            student_aligned: Aligned student probabilities [seq_len, student_vocab_size]
            teacher_aligned: Aligned teacher probabilities [seq_len, teacher_vocab_size]
        Returns:
            Combined hybrid loss
        """
        device = student_aligned.device
        # seq_len = student_aligned.size(0)  # Unused variable
        student_vocab_size = student_aligned.size(-1)
        teacher_vocab_size = teacher_aligned.size(-1)

        # Convert sets to sorted tensors for indexing 
        if self._teacher_matched_ids:
            teacher_matched_indices = torch.tensor(
                sorted(self._teacher_matched_ids), dtype=torch.long, device=device
            )
            teacher_indices_cpu = teacher_matched_indices.cpu().tolist()
            
            student_matched_indices = torch.tensor(
                [self._vocab_mapping[tid] for tid in teacher_indices_cpu], dtype=torch.long, device=device
            )
        else:
            teacher_matched_indices = torch.tensor([], dtype=torch.long, device=device)
            student_matched_indices = torch.tensor([], dtype=torch.long, device=device)

        # Create masks for unmatched tokens
        teacher_matched_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(student_vocab_size, dtype=torch.bool, device=device)

        if len(teacher_matched_indices) > 0:
            teacher_matched_mask[teacher_matched_indices] = True
            student_matched_mask[student_matched_indices] = True

        # 1. JSD loss for matched vocabulary tokens (direct semantic correspondence)
        matched_loss = torch.tensor(0.0, device=device)
        matched_token_count = 0
        if len(teacher_matched_indices) > 0:
            # Extract probabilities for matched tokens
            teacher_matched_probs = teacher_aligned[
                :, teacher_matched_indices
            ]  # [seq_len, num_matched]
            student_matched_probs = student_aligned[
                :, student_matched_indices
            ]  # [seq_len, num_matched]
            matched_token_count = teacher_matched_probs.size(-1)

            # Use JSD loss for semantically aligned tokens
            # Convert probabilities back to logits for JSD computation

            # Apply generalized JSD loss to matched tokens
            matched_loss = self._compute_jsd_loss_for_matched_tokens(
                student_matched_probs, teacher_matched_probs
            )

        # 2. Sorted comparison loss for unmatched vocabulary tokens
        teacher_unmatched_mask = ~teacher_matched_mask
        student_unmatched_mask = ~student_matched_mask

        teacher_unmatched_probs = teacher_aligned[
            :, teacher_unmatched_mask
        ]  # [seq_len, num_teacher_unmatched]
        student_unmatched_probs = student_aligned[
            :, student_unmatched_mask
        ]  # [seq_len, num_student_unmatched]

        unmatched_loss = torch.tensor(0.0, device=device)
        if teacher_unmatched_probs.size(-1) > 0 and student_unmatched_probs.size(-1) > 0:
            # Sort unmatched probabilities
            teacher_unmatched_sorted = teacher_unmatched_probs.sort(dim=-1, descending=True).values
            student_unmatched_sorted = student_unmatched_probs.sort(dim=-1, descending=True).values

            # Pad to same size if needed
            teacher_unmatched_size = teacher_unmatched_sorted.size(-1)
            student_unmatched_size = student_unmatched_sorted.size(-1)
            max_unmatched_size = max(teacher_unmatched_size, student_unmatched_size)

            if teacher_unmatched_size < max_unmatched_size:
                teacher_unmatched_sorted = F.pad(
                    teacher_unmatched_sorted, (0, max_unmatched_size - teacher_unmatched_size)
                )
            if student_unmatched_size < max_unmatched_size:
                student_unmatched_sorted = F.pad(
                    student_unmatched_sorted, (0, max_unmatched_size - student_unmatched_size)
                )

            # L1 loss on sorted unmatched tokens
            unmatched_loss = F.l1_loss(
                student_unmatched_sorted, teacher_unmatched_sorted, reduction="sum"
            )
            unmatched_loss /= student_aligned.size(0)  # Normalize by sequence length

        # 3. Combine losses with weights
        if self.hybrid_matched_weight is None:
            # Use adaptive weighting based on vocabulary overlap
            hybrid_matched_weight = matched_token_count / max(1, teacher_vocab_size)
            hybrid_unmatched_weight = 1.0 - hybrid_matched_weight
        else:
            # Use fixed weights provided in config
            hybrid_matched_weight = self.hybrid_matched_weight
            hybrid_unmatched_weight = self.hybrid_unmatched_weight

        total_loss = hybrid_matched_weight * matched_loss + hybrid_unmatched_weight * unmatched_loss

        # Store matched/unmatched components for logging
        self.last_matched_loss = matched_loss
        self.last_unmatched_loss = unmatched_loss

        return total_loss

    def _compute_jsd_loss_for_matched_tokens(self, student_logits, teacher_logits):
        """
        Compute JSD loss for matched vocabulary tokens.

        Args:
            student_logits: Student logits for matched tokens [seq_len, num_matched]
            teacher_logits: Teacher logits for matched tokens [seq_len, num_matched]
        Returns:
            JSD loss for matched tokens
        """
        # Reshape to [batch_size * seq_len, vocab_size] format expected by generalized_jsd_loss
        batch_seq_len, num_matched = student_logits.shape

        student_logits_reshaped = student_logits.view(-1, num_matched)
        teacher_logits_reshaped = teacher_logits.view(-1, num_matched)

        # Use the GOLD generalized JSD loss implementation that accepts probability inputs
        jsd_loss = GOLDTrainer.generalized_jsd_loss(
            student_logits_reshaped,
            teacher_logits_reshaped,
            labels=None,  # No masking needed for matched tokens
            beta=self.beta,  # Standard JSD beta
            temperature=1.0,  # Already applied in main computation
            reduction="batchmean",
            logits_are_probs=True,
        )

        return jsd_loss

    def _get_start_and_size_answers(self, answer_tensors):
        answers_index = []
        answers_size = []

        for answer in answer_tensors:
            answer_mask = answer.ne(self.ignore_index)
            if not answer_mask.any():
                answers_index.append(0)
                answers_size.append(0)
                continue

            valid_indices = answer_mask.nonzero(as_tuple=True)[0]
            answers_index.append(int(valid_indices[0].item()))
            answers_size.append(int(answer_mask.sum().item()))
        return answers_index, answers_size


class GOLDVLLMSyncCallback(TrainerCallback):
    """Sync the model weights to vLLM after training steps when it's safe to do so."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            # Check if this is a step where gradients are synchronized
            # This happens at the end of gradient accumulation cycles
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class GOLDTrainer(SFTTrainer):
    _tag_names = ["trl", "gold"]
    _name = "GOLD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str = None,
        changed_teacher_tokenizer: str| None = None, 
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        | None = None,
        peft_config: Optional["PeftConfig"] = None,
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        # Respect a user-provided data_collator; otherwise, provide a ChatML collator that
        if data_collator is None:
            data_collator = DataCollatorForChatML(
                tokenizer=processing_class, max_length=args.max_length
            )

        # Liger fused GKD loss (JSD)
        self.use_liger_gkd_loss = False
        if args.use_liger_kernel:
            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=args.beta,
                ignore_index=-100,
                temperature=args.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the GOLDConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["torch_dtype"] = (
                teacher_model_init_kwargs["torch_dtype"]
                if teacher_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["torch_dtype"])
            )

        if args.use_uld_loss and args.teacher_tokenizer_name_or_path is None:
            if isinstance(teacher_model, str):
                args.teacher_tokenizer_name_or_path = teacher_model
            else:
                raise ValueError(
                    "`teacher_tokenizer_name_or_path` must be set when using ULD loss with a pre-instantiated teacher model."
                )

        if isinstance(teacher_model, str):
            init_kwargs = dict(teacher_model_init_kwargs)
            if "torch_dtype" in init_kwargs and "dtype" not in init_kwargs:
                init_kwargs["dtype"] = init_kwargs.pop("torch_dtype")
            teacher_model = create_model_from_path(teacher_model, **init_kwargs)
        self.use_uld_loss = args.use_uld_loss
        self.teacher_tokenizer = None
        if args.use_uld_loss and args.teacher_tokenizer_name_or_path is not None:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                args.teacher_tokenizer_name_or_path
            )
            if (
                not hasattr(self.teacher_tokenizer, "pad_token")
                or self.teacher_tokenizer.pad_token is None
            ):
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        
        if getattr(args, "uld_use_hybrid_loss", False):
            if changed_teacher_tokenizer is None:
                raise ValueError("changed_teacher_tokenizer must be provided")
            self.changed_teacher_tokenizer = AutoTokenizer.from_pretrained(
                changed_teacher_tokenizer)
        else:
            self.changed_teacher_tokenizer = None

        # Hybrid ULD loss configuration is handled in ULDLoss class

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)
        if not args.use_uld_loss:
            teacher_model.resize_token_embeddings(self.model.config.vocab_size)

        # self.teacher_model = teacher_model 
        self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        # Hybrid ULD matched/unmatched accumulators (logged every step when ULD hybrid is used)
        self._matched_sum = 0.0
        self._unmatched_sum = 0.0
        self._matched_step_eq = 0.0
        self._unmatched_step_eq = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False

        self.uld_loss_fn = None
        if self.use_uld_loss:
            self.uld_loss_fn = ULDLoss(
                config=args,
                student_tokenizer=processing_class,
                teacher_tokenizer=self.teacher_tokenizer,
                changed_teacher_tokenizer=self.changed_teacher_tokenizer,
            )

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = (
            self.accelerator.num_processes
            * args.per_device_train_batch_size
            * args.steps_per_generation
        )
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Make sure tensor_parallel_size divides world size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(
                                self.accelerator.num_processes // self.vllm_tensor_parallel_size
                            )
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_structured_outputs_regex = args.vllm_structured_outputs_regex 
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))
            

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            # "prompts",
            # "prompt_attention_mask",
            # "messages",
            # "chat_template_kwargs",
            # "tools",
            # "original_prompt_text",
            # "original_completion_text",
            "messages",
            "nums",
            "target",
            "new_input_ids",
            "new_attention_mask",
            "new_labels",
            "teacher_input_ids",
            "original_completion_text",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
    ):

        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling to logits before computing probabilities
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature
            # Compute log probabilities for student and probabilities for teacher
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(
                beta, dtype=student_log_probs.dtype, device=student_log_probs.device
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]
                ),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(
                mixture_log_probs, teacher_log_probs, reduction="none", log_target=True
            )
            kl_student = F.kl_div(
                mixture_log_probs, student_log_probs, reduction="none", log_target=True
            )

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd
    
    def split_one_student_token(self,sid, student_tok, teacher_tok, preserve_ids):
        if sid in preserve_ids:
            return [sid]

        text = student_tok.decode(
            [sid],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        teacher_ids = teacher_tok.encode(text, add_special_tokens=False)

        # teacher 中本来就是一个 token
        if len(teacher_ids) == 1:
            return [sid]

        split_ids = []

        # 尝试将 teacher 的每个 token 文本转回一个 student token
        for tid in teacher_ids:
            piece = teacher_tok.decode(
                [tid],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            piece_ids = student_tok.encode(piece, add_special_tokens=False)

            if len(piece_ids) != 1:
                return [sid]

            split_ids.append(piece_ids[0])

        # 必须保证拆分后文本完全没变
        rebuilt = student_tok.decode(
            split_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        return split_ids if rebuilt == text else [sid]

    def _build_alignment_groups_from_ids(self, student_token_ids, teacher_token_ids,student_tokenizer,teacher_tokenizer):

            def to_canonical_pieces(tok, ids):
                pieces = []
                prev = ""
                for k in range(len(ids)):
                    # IMPORTANT: Do NOT skip special tokens - we need to align them too
                    cur = tok.decode(
                        ids[: k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    # Extract the incremental addition (may include spaces/ZWJ/etc.)
                    pieces.append(cur[len(prev) :])
                    prev = cur
                return pieces

            s_pieces = to_canonical_pieces(student_tokenizer, student_token_ids)
            t_pieces = to_canonical_pieces(teacher_tokenizer, teacher_token_ids)

            i = j = 0
            s_buf = t_buf = ""
            s_group = []
            t_group = []
            s_groups = []
            t_groups = []

            def flush():
                if s_group and t_group:
                    s_groups.append(s_group.copy())
                    t_groups.append(t_group.copy())

            # Greedily accumulate pieces until substrings match, then flush
            while i < len(s_pieces) or j < len(t_pieces):
                if s_buf == t_buf and s_buf != "":
                    flush()
                    s_buf = t_buf = ""
                    s_group = []
                    t_group = []
                    continue

                if s_buf == "" and i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                    continue
                if t_buf == "" and j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                    continue

                if len(s_buf) <= len(t_buf):
                    if i < len(s_pieces):
                        s_buf += s_pieces[i]
                        s_group.append(i)
                        i += 1
                    elif j < len(t_pieces):
                        t_buf += t_pieces[j]
                        t_group.append(j)
                        j += 1
                else:
                    if j < len(t_pieces):
                        t_buf += t_pieces[j]
                        t_group.append(j)
                        j += 1
                    elif i < len(s_pieces):
                        s_buf += s_pieces[i]
                        s_group.append(i)
                        i += 1

            # Flush any remainder if both sides accumulated something
            if s_buf == t_buf and s_group and t_group:
                flush()
            elif s_group or t_group:
                # Handle remaining unmatched tokens by forcing a flush
                # This ensures both sides have the same number of alignment groups
                if s_group or t_group:
                    # Ensure both groups have content (even if empty list)
                    if not s_group:
                        s_group = []
                    if not t_group:
                        t_group = []
                    # Force flush even if buffers don't match
                    if s_group or t_group:
                        s_groups.append(s_group.copy() if s_group else [])
                        t_groups.append(t_group.copy() if t_group else [])

            return s_groups, t_groups


    def build_alignment_groups(self,
        text,
        student_tok,
        teacher_tok,
        split_cache,
        preserve_ids,
    ):
        original_s_ids = student_tok.encode(text, add_special_tokens=False)

        s_ids = []
        for sid in original_s_ids:
            if sid not in split_cache:
                split_cache[sid] = self.split_one_student_token(
                    sid,
                    student_tok,
                    teacher_tok,
                    preserve_ids,
                )
            s_ids.extend(split_cache[sid])

        t_ids = teacher_tok.encode(text, add_special_tokens=False)

        s_groups, t_groups = self._build_alignment_groups_from_ids(
            s_ids,
            t_ids,
            student_tok,
            teacher_tok,
        )

        return s_ids, t_ids, s_groups, t_groups


    def expand_student_tokens(self, student_ids, student_tok, teacher_tok):
        """Per student token → list of teacher token ids via decode + re-encode."""
        expand_map = []
        for sid in student_ids:
            s = student_tok.decode([sid], skip_special_tokens=False,clean_up_tokenization_spaces=False,)
            t_ids = teacher_tok.encode(s, add_special_tokens=False)
            if not t_ids:
                raise ValueError("current student token can not be converted to teacher token!")
            expand_map.append(t_ids)
        return expand_map

    def build_single_supersequence(
        self,
        student_ids: List[int],
        teacher_ids: List[int],
        s_groups: List[List[int]],
        t_groups: List[List[int]],
        expand_map: List[List[int]],
        max_seq_len: int = 4096,
        prefix_ids: int = 4096,
    ) -> Tuple[List[int], Optional[torch.Tensor], List[int], List[int], List[int]]:
        
        prefix_len = len(prefix_ids)
        super_tokens = list(prefix_ids)
        position_ids = list(range(prefix_len))
        tok_group = [-1] * prefix_len
        tok_is_expand = [False] * prefix_len
        tok_owner_si = [-1] * prefix_len
        tok_ingroup_pos = [-1] * prefix_len
        tok_expand_offset = [-1] * prefix_len
        # ------------------------------------------------------------
        # 0. Basic maps
        # ------------------------------------------------------------
        s2g = {}
        s2p = {}
        for gi, sg in enumerate(s_groups):
            for p, si in enumerate(sg):
                s2g[si] = gi
                s2p[si] = p

        num_groups = len(s_groups)
        num_student_tokens = len(student_ids)

        # teacher_prefix_lens[gi] = number of original teacher tokens before group gi.
        teacher_prefix_lens = []
        cur_teacher_len = 0
        for gi in range(num_groups):
            teacher_prefix_lens.append(cur_teacher_len)
            cur_teacher_len += len(t_groups[gi])

        # ------------------------------------------------------------
        # 1. Build P block: all original Qwen teacher tokens
        # ------------------------------------------------------------
        # super_tokens = []
        # position_ids = []

        # tok_group = []          # group index for each physical token
        # tok_is_expand = []      # False for P block, True for E block
        # tok_owner_si = []       # student index if E token, -1 if P token
        # tok_ingroup_pos = []    # position inside s_group if E token, -1 if P token
        # tok_expand_offset = []  # offset inside current expanded student token, -1 if P token

        # P positions are original Qwen positions: 0, 1, 2, ...
        # P block is physically and logically the full original Qwen sequence.
        for gi in range(num_groups):
            for local_t_pos, ti in enumerate(t_groups[gi]):
                super_tokens.append(teacher_ids[ti])

                # Since t_groups are built from teacher_ids order,
                # teacher_prefix_lens[gi] + local_t_pos is the original teacher position.
                position_ids.append(prefix_len + teacher_prefix_lens[gi] + local_t_pos)

                tok_group.append(gi)
                tok_is_expand.append(False)
                tok_owner_si.append(-1)
                tok_ingroup_pos.append(-1)
                tok_expand_offset.append(-1)

        p_block_len = len(super_tokens)

        # ------------------------------------------------------------
        # 2. Build E block: all expanded Llama-token pieces, in original student order
        # ------------------------------------------------------------
        extract_positions = {}

        # For each group, we need to know how many expanded Qwen tokens appeared
        # before each student token inside this group.
        #
        # This is used for logical position_ids:
        #   pos(E(si, off)) =
        #       teacher_prefix_len_before_group
        #       + expanded_len_before_si_inside_group
        #       + off
        for gi in range(num_groups):
            expanded_len_so_far_in_group = 0

            for si in s_groups[gi]:
                if si < 0 or si >= num_student_tokens:
                    continue

                p_in_g = s2p[si]
                expanded = expand_map[si]

                if not expanded:
                    continue

                for off, tok_id in enumerate(expanded):
                    super_tokens.append(tok_id)

                    logical_pos = (
                        prefix_len 
                        + teacher_prefix_lens[gi]
                        + expanded_len_so_far_in_group
                        + off
                    )
                    position_ids.append(logical_pos)

                    tok_group.append(gi)
                    tok_is_expand.append(True)
                    tok_owner_si.append(si)
                    tok_ingroup_pos.append(p_in_g)
                    tok_expand_offset.append(off)

                # Extract from the last token of this student's expanded piece.
                extract_positions[si] = len(super_tokens) - 1

                expanded_len_so_far_in_group += len(expanded)

        L = len(super_tokens)

        if L == 0:
            return [], None, [], [], []

        # Physical length guard.
        #
        # Your original code used max_seq_len * 2 because supersequence contains
        # both original teacher tokens and expanded tokens.
        if L > max_seq_len * 2.5:
            return [], None, [], [], []

        # Logical position guard.
        #
        # Even if physical length is okay, if logical RoPE position exceeds max_seq_len,
        if max(position_ids) + 1 > max_seq_len *2 :
            return [], None, [], [], []

        # ------------------------------------------------------------
        # 3. Vectorized attention mask
        # ------------------------------------------------------------
        t_group = torch.tensor(tok_group, dtype=torch.long)
        t_is_exp = torch.tensor(tok_is_expand, dtype=torch.bool)
        t_is_pfx = ~t_is_exp
        t_igp = torch.tensor(tok_ingroup_pos, dtype=torch.long)
        t_owner_si = torch.tensor(tok_owner_si, dtype=torch.long)

        gi_q = t_group.unsqueeze(1)     # (L, 1)
        gi_k = t_group.unsqueeze(0)     # (1, L)

        is_exp_q = t_is_exp.unsqueeze(1)
        is_pfx_q = t_is_pfx.unsqueeze(1)

        is_exp_k = t_is_exp.unsqueeze(0)
        is_pfx_k = t_is_pfx.unsqueeze(0)

        igp_q = t_igp.unsqueeze(1)
        igp_k = t_igp.unsqueeze(0)

        owner_q = t_owner_si.unsqueeze(1)
        owner_k = t_owner_si.unsqueeze(0)

        pos = torch.arange(L)
        causal = pos.unsqueeze(1) >= pos.unsqueeze(0)

        # ------------------------------------------------------------
        # Rule A: P query tokens
        #
        # P block is the full original Qwen sequence.
        # Prefix tokens should run as a normal causal language-model sequence
        # inside P block only.
        #
        # They should not attend to E block.
        # ------------------------------------------------------------
        pfx_rule = is_pfx_k & causal

        # ------------------------------------------------------------
        # Rule B: E query tokens
        #
        # Important:
        #   E query should NOT see:
        #       - P tokens from the same group
        #       - E tokens from previous groups
        #       - E tokens from future groups
        # ------------------------------------------------------------
        e_rule = (
            ((gi_k < gi_q) & is_pfx_k)
            | ((gi_k == gi_q) & is_exp_k & (igp_k < igp_q))
            | ((gi_k == gi_q) & is_exp_k & (owner_k == owner_q) & causal)
        )

        attn_mask = torch.where(is_pfx_q, pfx_rule, e_rule)

        # ------------------------------------------------------------
        # 4. Collect extraction positions
        # ------------------------------------------------------------
        extract_pos_list = []
        valid_si_list = []

        for si in range(num_student_tokens):
            if si in extract_positions:
                extract_pos_list.append(extract_positions[si])
                valid_si_list.append(si)

        return super_tokens, attn_mask, position_ids, extract_pos_list, valid_si_list
    
    def get_teacher_logits(
        self,
        teacher_model,
        teacher_prompt_ids,
        student_completion_ids,
        student_tok,
        teacher_tok,
        device,
    ):
        # 1. student completion -> text -> teacher token
        text = student_tok.decode(
            student_completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        teacher_completion_ids = teacher_tok.encode(text, add_special_tokens=False)

        # 2. alignment
        sg, tg = self._build_alignment_groups_from_ids(
            student_completion_ids,
            teacher_completion_ids,
            student_tok,
            teacher_tok,
        )

        # 3. student token for teacher token
        em = self.expand_student_tokens(
            student_completion_ids,
            student_tok,
            teacher_tok,
        )

        # 4. P + E
        super_ids, attn_mask, position_ids, extract_pos, _ = self.build_single_supersequence(
            student_completion_ids,
            teacher_completion_ids,
            sg,
            tg,
            em,
            max_seq_len=4096,
            prefix_ids=teacher_prompt_ids,
        )

        input_ids = torch.tensor([super_ids], device=device)
        position_ids = torch.tensor([position_ids], device=device)

        attn_mask = attn_mask[None, None].to(device)
        attn_mask = torch.where(
            attn_mask,
            torch.tensor(0.0, device=device),
            torch.tensor(float("-inf"), device=device),
        )

        with torch.no_grad():
            outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=position_ids,
                use_cache=False,
            )

        return outputs.logits[0, extract_pos, :]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        assert self.teacher_tokenizer
        assert self.teacher_tokenizer.eos_token is not None
        assert self.uld_loss_fn

        outputs_student = model(
            input_ids=inputs["new_input_ids"],
            attention_mask=inputs["new_attention_mask"],
            use_cache=False,
        )

        student_labels = inputs["new_labels"]

        losses = []
        matched_losses = []
        unmatched_losses = []
        for i in range(outputs_student.logits.shape[0]):

            # completion所在位置
            valid = student_labels[i] != -100
            pos = valid.nonzero(as_tuple=True)[0]

            if len(pos) == 0:
                continue

            start = pos[0].item()

            student_completion_ids = inputs["new_input_ids"][i, valid].tolist()

            teacher_prompt_ids = inputs["teacher_input_ids"][i]
            teacher_prompt_ids = teacher_prompt_ids[
                teacher_prompt_ids != self.teacher_tokenizer.pad_token_id
            ].tolist()

            teacher_logits = self.get_teacher_logits(
                self.teacher_model,
                teacher_prompt_ids,
                student_completion_ids,
                self.processing_class,      # student tokenizer
                self.teacher_tokenizer,     # Qwen tokenizer
                self.accelerator.device,
            )
            
        
            n = len(student_completion_ids)

            student_logits = outputs_student.logits[i, start:start + n, :]

            if self.uld_loss_fn.skip_student_eos:
                student_logits = student_logits[:-1]
            if self.uld_loss_fn.skip_teacher_eos:
                teacher_logits = teacher_logits[:-1]
            
            if student_logits.size(0) == 0 or teacher_logits.size(0) == 0:
                losses.append(outputs_student.logits[i].sum() * 0.0)
                continue
            assert student_logits.size(0) == teacher_logits.size(0)

            student_probs = F.softmax(student_logits / self.uld_loss_fn.student_temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.uld_loss_fn.teacher_temperature, dim=-1)

            loss_i = self.uld_loss_fn._compute_hybrid_uld_loss(
                student_probs,
                teacher_probs,
            )

            losses.append(loss_i)
            matched_losses.append(self.uld_loss_fn.last_matched_loss.detach())
            unmatched_losses.append(self.uld_loss_fn.last_unmatched_loss.detach())

        loss = torch.stack(losses).mean()

        matched_val = torch.stack(matched_losses).mean().item()
        unmatched_val = torch.stack(unmatched_losses).mean().item()

        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_eq = 1.0 / ga

        self._matched_sum += matched_val * step_eq
        self._unmatched_sum += unmatched_val * step_eq
        self._matched_step_eq += step_eq
        self._unmatched_step_eq += step_eq

        empty_cache()

        return (loss, outputs_student) if return_outputs else loss

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id):
        device = self.accelerator.device
        input_ids = cast(torch.Tensor, inputs["input_ids"])
        attention_masks = cast(torch.Tensor, inputs["attention_masks"])
        prompt_length = input_ids.shape[1]

        # Generate output with respect to the prompt only
        generated_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            output_logits=True,
            generation_config=generation_config,
            return_dict_in_generate=True,
        )
        # Original prompt + student completion (padded both sides)
        new_input_ids = generated_outputs.sequences

        # Mask out padding in `new_input_ids`
        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_attention_mask[new_input_ids == pad_token_id] = 0

        # Completion tokens. Mask out everything else.
        new_labels = new_input_ids.clone()
        new_labels[new_input_ids == pad_token_id] = -100
        new_labels[:, :prompt_length] = -100

        new_labels_ragged = [row[row != -100] for row in new_labels]

        prompt_token_ids = [
            prompt.masked_select(prompt != pad_token_id).tolist() for prompt in input_ids
        ]
        prompt_texts = self.processing_class.batch_decode(prompt_token_ids)
        # Remove EOS token
        completion_texts = self.processing_class.batch_decode(
            new_labels_ragged, skip_special_tokens=True
        )

        return new_input_ids, new_attention_mask, new_labels, prompt_texts, completion_texts

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):

        device = self.accelerator.device
        

        # Decode prompts for vLLM (without special tokens - vLLM expects clean text)
        # print("PROMPTS", inputs['input_ids'])
        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs['input_ids'],
            skip_special_tokens=True,
            # clean_up_tokenization_spaces=False # Keep this commented unless specific issues arise
        )
        # print("PROMPTS_TEXT_FOR_VLLM", prompts_text_for_vllm)
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm
            ]

        # Also decode prompts WITH special tokens for ULD loss computation
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs['input_ids'], 
            skip_special_tokens=False,
        )

        # system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        # target_system_prompt = "You are a helpful assistant."
        # prompts_text = [p.replace(target_system_prompt, system_prompt) for p in prompts_text]
        # Add system prompt to prompts

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = (
            generation_config.top_k
            if generation_config.top_k and generation_config.top_k > 0
            else -1
        )
        # top_p, repetition_penalty, min_p are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = (
            self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        )
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion per prompt from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    structured_outputs_regex=self.vllm_structured_outputs_regex,
                )["completion_ids"]
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(
                    gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group
                )
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [
                output.token_ids for outputs in all_outputs for output in outputs.outputs
            ]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(
                    local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size
                )
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Use prompts_text_for_vllm (without special tokens) for tokenization since vLLM expects clean text
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        # Calculate max_length for prompts, ensuring it's positive
        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,),
                            pad_token_id,
                            device=device,
                            dtype=completion_tensor.dtype,
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Mask prompt tokens in labels
        prompt_lengths = prompt_ids.shape[1]
        new_labels[:, :prompt_lengths] = -100

        # IMPORTANT: Preserve original text for cross-tokenizer ULD loss
        # Use prompts_text_with_special (with special tokens) for ULD loss computation
        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return (
            new_input_ids,
            new_attention_mask,
            new_labels,
            prompts_text_with_special,
            completion_texts,
        )

    @profiling_decorator
    def generate_on_policy_outputs_vllm(
        self,
        inputs: dict[str, torch.Tensor],
        generation_config: GenerationConfig,
        pad_token_id: int,
    ):
        assert self.vllm_mode == "colocate"
        device = self.accelerator.device

        # Construct vLLM input (remove padding)
        padded_prompt_ids = inputs["input_ids"].cpu()
        prompt_token_ids = [
            prompt.masked_select(prompt != pad_token_id).tolist() for prompt in padded_prompt_ids
        ]

        sampling_params = SamplingParams(
            repetition_penalty=getattr(self.args, "repetition_penalty", 1.0),
            temperature=generation_config.temperature,
            top_p=getattr(self.args, "top_p", 1.0),
            top_k=generation_config.top_k
            if generation_config.top_k and generation_config.top_k > 0
            else -1,
            min_p=getattr(self.args, "min_p", 0.0),
            max_tokens=generation_config.max_new_tokens,
        )

        outputs = self.vllm_engine.generate(
            prompts=[{"prompt_token_ids": x} for x in prompt_token_ids],
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        completion_ids = [
            torch.tensor(output.outputs[0].token_ids, dtype=torch.int64) for output in outputs
        ]
        padded_completion_ids = torch.nn.utils.rnn.pad_sequence(
            completion_ids, batch_first=True, padding_value=pad_token_id
        )

        if self.vllm_enable_sleep_mode:
            self.vllm_engine.sleep(level=2)

        # Original prompt + student completion (padded both sides)
        new_input_ids = torch.cat([padded_prompt_ids, padded_completion_ids], dim=1).to(device)

        # 1 for prompt/completion, 0 for padding
        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_attention_mask[new_input_ids == pad_token_id] = 0

        # Completion tokens + -100 everywhere else
        new_labels = new_input_ids.clone()
        new_labels[new_input_ids == pad_token_id] = -100
        new_labels[:, : padded_prompt_ids.shape[1]] = -100

        prompt_texts = self.processing_class.batch_decode(prompt_token_ids)
        completion_texts = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        return new_input_ids, new_attention_mask, new_labels, prompt_texts, completion_texts

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # recurse into the child
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                # use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    @profiling_decorator
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step for the General Online Logit Distillation (GOLD) model.

        This method implements the on-policy learning approach described in the GOLD blog post. With probability
        `self.lmbda`, it generates new responses using the student model, which are then used for training instead of
        the offline original inputs.
        """
        on_policy = False
        if random.random() <= self.lmbda:
            on_policy = True
            if self.use_vllm:
                self._wake_vllm_if_needed()
                result = self._generate_on_policy_outputs_vllm(
                    inputs, self.generation_config, self.processing_class.pad_token_id
                )
                new_input_ids, new_attention_mask, new_labels, prompt_texts, completion_texts = (
                    result
                )
            else:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    result = self.generate_on_policy_outputs(
                        unwrapped_model,
                        inputs,
                        self.generation_config,
                        self.processing_class.pad_token_id,
                    )
                    (
                        new_input_ids,
                        new_attention_mask,
                        new_labels,
                        prompt_texts,
                        completion_texts,
                    ) = result

            inputs["new_input_ids"] = new_input_ids
            inputs["new_attention_mask"] = new_attention_mask
            inputs["new_labels"] = new_labels

            # CRITICAL: Preserve original text for cross-tokenizer ULD loss
            # This ensures both off-policy (dataset) and on-policy (generated) samples
            # can use proper text-based alignment for different tokenizers
            inputs["original_completion_text"] = completion_texts

            # Log prompt and completion texts
            self._textual_logs["prompt"].extend(gather_object(prompt_texts))
            self._textual_logs["completion"].extend(gather_object(completion_texts))

        loss = super().training_step(model, inputs, num_items_in_batch)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", ["past_key_values"]
                )
            else:
                ignore_keys = []

        labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
        if len(labels) == 1:
            labels = labels[0]

        with torch.no_grad():
            with self.compute_loss_context_manager():
                input_ids, attention_mask = inputs["input_ids"], inputs["attention_masks"]
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_logits=False,
                )
                
                
                
                prompt_length = input_ids.shape[1]
                completion_ids = outputs.sequences[:, prompt_length:]
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]
        completion_ids = nested_detach(completion_ids)

        return None, completion_ids, labels

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        if mode == "train":
            device = (
                self.accelerator.device
                if hasattr(self.accelerator, "device")
                else torch.device("cpu")
            )
            # include matched/unmatched accumulators for distributed reduction
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                    self._matched_sum,
                    self._unmatched_sum,
                    self._matched_step_eq,
                    self._unmatched_step_eq,
                ],
                dtype=torch.float64,
                device=device,
            )

            # Sum across processes so we mirror Trainer's distributed reduction
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO)
                != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
                matched_sum,
                unmatched_sum,
                matched_eq,
                unmatched_eq,
            ) = vec.tolist()

            # Compute category averages over the *same window* as Trainer's logs
            # (avoid div-by-zero if, e.g., no on-policy steps in the window)
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # matched/unmatched averaged over same logging window (if present)
            if matched_eq > 0:
                logs["matched_loss"] = round(matched_sum / matched_eq, 4)
            if unmatched_eq > 0:
                logs["unmatched_loss"] = round(unmatched_sum / unmatched_eq, 4)

            # Reset window accumulators after logging (just like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0
            self._matched_sum = self._unmatched_sum = 0.0
            self._matched_step_eq = self._unmatched_step_eq = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):
            if is_rich_available():
                print_prompt_completions_sample_uld(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=self.num_completions_to_print, random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)})
