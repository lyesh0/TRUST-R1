# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import math
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        temperature = self._finite_number('temperature', temperature)
        if temperature <= 0:
            raise ValueError(f'temperature must be positive, got {temperature}')
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        bad_grad = self._sync_flag(not torch.isfinite(grad_norm).all().item(), grad_norm.device)
        if bad_grad:
            self.actor_optimizer.zero_grad()
            return grad_norm, True
        self.actor_optimizer.step()
        return grad_norm, False

    @staticmethod
    def _has_non_finite(*tensors):
        for tensor in tensors:
            if tensor is not None and not torch.isfinite(tensor).all().item():
                return True
        return False

    @staticmethod
    def _finite_number(name, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be numeric, got {value!r}')
        if not math.isfinite(number):
            raise ValueError(f'{name} must be finite, got {value!r}')
        return number

    def _validate_update_config(self, temperature):
        learning_rate = self._finite_number('optim.lr', self.config.optim.lr)
        grad_clip = self._finite_number('grad_clip', self.config.grad_clip)
        clip_ratio = self._finite_number('clip_ratio', self.config.clip_ratio)
        entropy_coeff = self._finite_number('entropy_coeff', self.config.entropy_coeff)
        temperature = self._finite_number('temperature', temperature)
        if learning_rate <= 0 or grad_clip <= 0 or temperature <= 0:
            raise ValueError('optim.lr, grad_clip, and temperature must be positive')
        if not 0 < clip_ratio < 1:
            raise ValueError(f'clip_ratio must be in (0, 1), got {clip_ratio}')
        if entropy_coeff < 0:
            raise ValueError(f'entropy_coeff must be non-negative, got {entropy_coeff}')
        if self.config.ppo_epochs != 1:
            raise ValueError('DataParallelPPOActor currently requires ppo_epochs=1')
        if self.config.use_kl_loss:
            kl_loss_coef = self._finite_number('kl_loss_coef', self.config.kl_loss_coef)
            if kl_loss_coef < 0:
                raise ValueError(f'kl_loss_coef must be non-negative, got {kl_loss_coef}')

    @staticmethod
    def _sync_flag(flag, device):
        """Return True on every rank when any rank reports the flag."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return bool(flag)
        flag_tensor = torch.tensor(int(bool(flag)), dtype=torch.int32, device=device)
        torch.distributed.all_reduce(flag_tensor, op=torch.distributed.ReduceOp.MAX)
        return bool(flag_tensor.item())

    @staticmethod
    def _log_ratio_metrics(log_ratio, mask, prefix):
        active = log_ratio.detach().masked_select(mask.bool())
        clamp = core_algos.EXPONENTIAL_LOG_RATIO_CLAMP
        return {
            f'actor/{prefix}_abs_max': active.abs().max().item(),
            f'actor/{prefix}_clamp_frac': (active.abs() > clamp).float().mean().item(),
        }

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        self._validate_update_config(temperature)

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            select_keys.append('loss_mask')
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()
            valid_micro_batches = 0
            skipped_zero_mask = 0
            skipped_non_finite = 0
            skip_optimizer_step = False

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if self.config.state_masking:
                    response_mask = data['loss_mask']
                local_mask_non_finite = self._has_non_finite(response_mask)
                mask_non_finite = self._sync_flag(local_mask_non_finite, response_mask.device)
                local_zero_mask = not local_mask_non_finite and response_mask.sum().item() <= 0
                zero_mask = self._sync_flag(local_zero_mask, response_mask.device)
                if mask_non_finite or zero_mask:
                    skipped_zero_mask += 1
                    skipped_non_finite += int(mask_non_finite)
                    skip_optimizer_step = True
                    break
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                ppo_log_ratio = log_prob - old_log_prob
                bad_policy_input = self._sync_flag(
                    self._has_non_finite(entropy, log_prob, old_log_prob, advantages, ppo_log_ratio),
                    log_prob.device)
                if bad_policy_input:
                    skipped_non_finite += 1
                    skip_optimizer_step = True
                    break
                ratio_metrics = self._log_ratio_metrics(ppo_log_ratio, response_mask, 'ppo_log_ratio')

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    ref_log_ratio = ref_log_prob - log_prob
                    bad_ref_input = self._sync_flag(self._has_non_finite(ref_log_prob, ref_log_ratio), log_prob.device)
                    if bad_ref_input:
                        skipped_non_finite += 1
                        skip_optimizer_step = True
                        break
                    ratio_metrics.update(self._log_ratio_metrics(ref_log_ratio, response_mask, 'ref_log_ratio'))
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                bad_loss = self._sync_flag(
                    self._has_non_finite(pg_loss, pg_clipfrac, ppo_kl, entropy_loss, policy_loss),
                    log_prob.device)
                if bad_loss:
                    skipped_non_finite += 1
                    skip_optimizer_step = True
                    break

                loss = policy_loss / self.gradient_accumulation
                loss.backward()
                valid_micro_batches += 1

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                data.update(ratio_metrics)
                append_to_dict(metrics, data)

            if skip_optimizer_step or valid_micro_batches == 0:
                self.actor_optimizer.zero_grad()
                data = {
                    'actor/grad_norm': 0.0,
                    'actor/update_skipped': 1.0,
                    'actor/skipped_zero_mask_micro_batches': skipped_zero_mask,
                    'actor/skipped_non_finite_micro_batches': skipped_non_finite,
                }
                append_to_dict(metrics, data)
                continue

            grad_norm, skipped_bad_grad = self._optimizer_step()
            data = {
                'actor/grad_norm': 0.0 if skipped_bad_grad else grad_norm.detach().item(),
                'actor/update_skipped': 1.0 if skipped_bad_grad else 0.0,
                'actor/skipped_zero_mask_micro_batches': skipped_zero_mask,
                'actor/skipped_non_finite_micro_batches': skipped_non_finite,
            }
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
