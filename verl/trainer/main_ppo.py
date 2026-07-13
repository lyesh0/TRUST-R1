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
from trust_r1.reward_adapter import compute_trust_reward
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

    def __init__(self, tokenizer, num_examine, format_score=0., trust_reward_config=None, trust_logging_config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.trust_reward_config = trust_reward_config
        self.trust_logging_config = trust_logging_config or {}
        self.last_trust_reward_metrics = {}
        self._trajectory_write_count = 0

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        self.last_trust_reward_metrics = {}
        trust_reward_items = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score)
            trace_summary = data_item.non_tensor_batch.get('trust_r1_trace_summary', {})
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

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        self.last_trust_reward_metrics = self._build_trust_reward_metrics(trust_reward_items)

        return reward_tensor

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
    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        trust_reward_config=trust_reward_config,
        trust_logging_config=trust_logging_config,
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=config.trainer.get('val_num_examine', 1),
        trust_reward_config=trust_reward_config,
        trust_logging_config=trust_logging_config,
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
