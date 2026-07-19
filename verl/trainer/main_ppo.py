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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from pathlib import Path
import json

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from trust_r1.config import RewardConfig
from trust_r1.process_reward import build_process_features
from trust_r1.reward_adapter import compute_trust_reward, extract_final_answer
import re
import numpy as np

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 format_score=0.,
                 trust_reward_config=None,
                 trust_logging_config=None,
                 process_reward_config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.trust_reward_config = trust_reward_config
        self.trust_logging_config = trust_logging_config or {}
        self.process_reward_config = process_reward_config or {}
        self.last_trust_reward_metrics = {}
        self.last_process_metrics = {}
        self.last_stage1_records = []
        self._trajectory_write_count = 0

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        self.last_trust_reward_metrics = {}
        self.last_process_metrics = {}
        self.last_stage1_records = []
        trust_reward_items = []
        process_features = []
        answer_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = int(data_item.batch['attention_mask'][:prompt_length].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = int(data_item.batch['attention_mask'][prompt_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
            answer_scores.append(float(score))
            trace_summary = data_item.non_tensor_batch.get('trust_r1_trace_summary', {})
            rollout_trace = data_item.non_tensor_batch.get('trust_r1_rollout_traces', {})

            compute_process = bool(self.process_reward_config.get('compute_diagnostics', False) or
                                   self.process_reward_config.get('enabled', False))
            features = None
            if compute_process:
                features = build_process_features(
                    response_token_ids=response_ids,
                    valid_response_length=valid_response_length,
                    gold_aliases=ground_truth.get('target', []),
                    rollout_trace=rollout_trace,
                    tokenizer=self.tokenizer,
                    max_search_steps=int(self.process_reward_config.get('max_search_steps', 2)),
                )
                process_features.append(features)
                if (self.process_reward_config.get('enabled', False)
                        and self.process_reward_config.get('abort_on_alignment_error', True)
                        and not features.alignment_valid):
                    sample_id = data_item.non_tensor_batch.get('index', i)
                    raise ValueError(f"Stage1 query/trace alignment failed for sample {sample_id}")
            trust_result = None
            if self.trust_reward_config is not None and self.trust_reward_config.get('enabled', False):
                trust_result = compute_trust_reward(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    had_fault=bool(trace_summary.get('had_fault', False)) if isinstance(trace_summary, dict) else False,
                    changed_query_after_fault=bool(trace_summary.get('changed_query_after_fault', False)) if isinstance(trace_summary, dict) else False,
                    config=RewardConfig.from_mapping(self.trust_reward_config),
                )
                score = trust_result.reward.total
                trust_reward_items.append(trust_result)
                self._maybe_write_trajectory(
                    data_item=data_item,
                    data_source=data_source,
                    sequences_str=sequences_str,
                    ground_truth=ground_truth,
                    trace_summary=trace_summary,
                    trust_result=trust_result,
                )

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = score

            response_str = self.tokenizer.decode(valid_response_ids)
            record = {
                'question_id': str(data_item.non_tensor_batch.get('index', i)),
                'prompt': self.tokenizer.decode(valid_prompt_ids),
                'gold_aliases': ground_truth.get('target', []),
                'queries': features.queries if features is not None else [],
                'information_blocks': features.information_blocks if features is not None else [],
                'evidence_hit_by_step': features.evidence_hits if features is not None else [],
                'first_hit_reward_by_step': features.step_rewards.detach().cpu().tolist() if features is not None else [],
                'alignment_valid': features.alignment_valid if features is not None else True,
                'parse_error_count': features.parse_error_count if features is not None else 0,
                'final_answer': extract_final_answer(response_str),
                'answer_correct': bool(answer_scores[-1] > 0),
                'answer_score': answer_scores[-1],
                'valid_action': bool(trace_summary.get('valid_action', True)) if isinstance(trace_summary, dict) else True,
                'invalid_action_count': int(trace_summary.get('invalid_action_count', 0)) if isinstance(trace_summary, dict) else 0,
                'finish_reason': trace_summary.get('finish_reason', 'max_turns') if isinstance(trace_summary, dict) else 'max_turns',
                'search_count': int(trace_summary.get('search_count', 0)) if isinstance(trace_summary, dict) else 0,
                'trust_r1_trace_summary': trace_summary if isinstance(trace_summary, dict) else {},
                'reward_breakdown': trust_result.reward.to_dict() if trust_result is not None else None,
                'response_text': response_str,
            }
            self.last_stage1_records.append(record)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        self.last_trust_reward_metrics = self._build_trust_reward_metrics(trust_reward_items)
        if process_features:
            data.batch['process_step_rewards'] = torch.stack([item.step_rewards for item in process_features])
            data.batch['query_step_ids'] = torch.stack([item.query_step_ids for item in process_features])
            data.batch['process_evidence_hits'] = torch.tensor(
                [item.evidence_hits for item in process_features],
                dtype=torch.bool,
                device=reward_tensor.device,
            )
            data.batch['process_alignment_valid'] = torch.tensor(
                [item.alignment_valid for item in process_features],
                dtype=torch.bool,
                device=reward_tensor.device,
            )
            search_count = sum(len(item.queries) for item in process_features)
            raw_hits = sum(sum(item.evidence_hits[:len(item.queries)]) for item in process_features)
            first_hits = sum(item.step_rewards[:len(item.queries)].sum().item() for item in process_features)
            self.last_process_metrics = {
                'process/raw_hit_rate': raw_hits / search_count if search_count else 0.0,
                'process/first_hit_rate': first_hits / search_count if search_count else 0.0,
                'process/span_alignment_error_count': float(sum(not item.alignment_valid for item in process_features)),
                'process/parse_error_count': float(sum(item.parse_error_count for item in process_features)),
                'answer/em': float(np.mean(answer_scores)) if answer_scores else 0.0,
            }

        return reward_tensor

    def write_stage1_records(self, *, split, step, local_advantages=None, query_step_ids=None, weight=0.2):
        if not self.trust_logging_config.get('enabled', False):
            return
        if not self.trust_logging_config.get('write_trajectories', False):
            return
        output_dir = self.trust_logging_config.get('output_dir')
        if not output_dir or not self.last_stage1_records:
            return

        if split == 'train':
            path = Path(output_dir) / 'train_trajectories.jsonl'
            limit = int(self.trust_logging_config.get('sample_limit_per_call', 32) or 0)
            records = self.last_stage1_records if limit < 0 else self.last_stage1_records[:limit]
        elif split == 'validation':
            path = Path(output_dir) / f'validation_step_{int(step)}.jsonl'
            records = self.last_stage1_records
        else:
            raise ValueError(f'unsupported Stage1 trajectory split: {split}')

        path.parent.mkdir(parents=True, exist_ok=True)
        rollout_counts = {}
        with path.open('a', encoding='utf-8') as f:
            for index, original in enumerate(records):
                record = dict(original)
                question_id = record['question_id']
                rollout_id = rollout_counts.get(question_id, 0)
                rollout_counts[question_id] = rollout_id + 1
                record['rollout_id'] = rollout_id
                record['trainer_step'] = int(step)
                record['local_z_by_step'] = []
                if local_advantages is not None and query_step_ids is not None and weight > 0:
                    max_step = len(record.get('first_hit_reward_by_step', []))
                    for step_id in range(1, max_step + 1):
                        mask = query_step_ids[index] == step_id
                        local_sum = local_advantages[index, mask].sum().item() if mask.any().item() else 0.0
                        record['local_z_by_step'].append(local_sum / weight)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def reset_stage1_validation_records(self, step):
        if not self.trust_logging_config.get('enabled', False):
            return
        output_dir = self.trust_logging_config.get('output_dir')
        if not output_dir:
            return
        path = Path(output_dir) / f'validation_step_{int(step)}.jsonl'
        if path.exists():
            path.unlink()

    def _build_trust_reward_metrics(self, trust_reward_items):
        if not trust_reward_items:
            return {}

        def mean(values):
            return float(np.mean(values)) if values else 0.0

        reward_dicts = [item.reward.to_dict() for item in trust_reward_items]
        return {
            'trust_r1_reward/answer_mean': mean([item['answer'] for item in reward_dicts]),
            'trust_r1_reward/format_mean': mean([item['format'] for item in reward_dicts]),
            'trust_r1_reward/recovery_mean': mean([item['recovery'] for item in reward_dicts]),
            'trust_r1_reward/duplicate_penalty_mean': mean([item['duplicate_penalty'] for item in reward_dicts]),
            'trust_r1_reward/invalid_penalty_mean': mean([item['invalid_penalty'] for item in reward_dicts]),
            'trust_r1_reward/total_mean': mean([item['total'] for item in reward_dicts]),
            'trust_r1_reward/answer_correct_rate': mean([float(item.answer_correct) for item in trust_reward_items]),
            'trust_r1_reward/evidence_recovered_rate': mean([float(item.evidence_recovered) for item in trust_reward_items]),
            'trust_r1_reward/duplicate_query_count_mean': mean([item.duplicate_query_count for item in trust_reward_items]),
        }

    def _maybe_write_trajectory(self, *, data_item, data_source, sequences_str, ground_truth, trace_summary, trust_result):
        if not self.trust_logging_config.get('enabled', False):
            return
        if not self.trust_logging_config.get('write_trajectories', False):
            return
        limit = int(self.trust_logging_config.get('sample_limit_per_call', 32) or 0)
        if limit >= 0 and self._trajectory_write_count >= limit:
            return
        output_dir = self.trust_logging_config.get('output_dir')
        if not output_dir:
            return

        path = Path(output_dir) / 'trajectories.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        reward_model = data_item.non_tensor_batch.get('reward_model', {})
        sample_id = data_item.non_tensor_batch.get('index', self._trajectory_write_count)
        record = {
            'sample_id': str(sample_id),
            'data_source': str(data_source),
            'gold_answer': ground_truth.get('target') if isinstance(ground_truth, dict) else ground_truth,
            'final_answer': trust_result.parsed.answer,
            'is_correct': trust_result.answer_correct,
            'search_queries': trust_result.parsed.search_queries,
            'information_block_count': len(trust_result.parsed.information_blocks),
            'information_previews': [block[:240] for block in trust_result.parsed.information_blocks[:3]],
            'trust_r1_trace_summary': trace_summary if isinstance(trace_summary, dict) else {},
            'reward_breakdown': trust_result.reward.to_dict(),
            'reward_model': reward_model,
            'solution_preview': sequences_str[-2000:],
        }
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._trajectory_write_count += 1


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    trust_reward_config = config.get('trust_reward', None)
    trust_logging_config = config.get('trust_r1_logging', None)
    process_reward_config = config.get('process_reward', None)
    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        trust_reward_config=trust_reward_config,
        trust_logging_config=trust_logging_config,
        process_reward_config=process_reward_config,
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=config.trainer.get('val_num_examine', 1),
        trust_reward_config=trust_reward_config,
        trust_logging_config=trust_logging_config,
        process_reward_config=process_reward_config,
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
