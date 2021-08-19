# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical

from fairseq import utils
from fairseq.criterions import FairseqSequenceCriterion, register_criterion

class BaselineEstimator(nn.Module):

    def __init__(self, input_size):
        super(BaselineEstimator, self).__init__()
        self.ff1 = nn.Linear(input_size, input_size * 4)
        self.ff2 = nn.Linear(input_size * 4, 1)
        
    def forward(self, input, mean=False):

        input = input.detach()

        if mean:
            input = input.mean(axis=0)
        out = self.ff1(input)
        out = F.relu(out)
        out = self.ff2(out)
        return out

@register_criterion('sequence_reinforce')
class SequenceReinforceCriterion(FairseqSequenceCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        from fairseq.tasks.translation_struct import TranslationStructuredPredictionTask
        if not isinstance(task, TranslationStructuredPredictionTask):
            raise Exception(
                'sequence_risk criterion requires `--task=translation_struct`'
            )

        self.baseline = BaselineEstimator(args.encoder_embed_dim)
        self.baseline.half()
        self.baseline_optimizer = optim.SGD(self.baseline.parameters(), lr=0.01)
        self.sentence_level_reward = args.sentence_level_reward
        self.use_baseline_reward = args.baseline_reward
        self.optimize_baseline_reward = args.optimize_baseline_reward
        self.a2c = args.a2c

        self.num_warmup_steps = args.baseline_warmup_steps
        self.num_steps = 0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--normalize-costs', action='store_true',
                            help='normalize costs within each hypothesis')

        # fmt: on

    def forward(self, model, sample, reduce=True, use_enc_for_baseline=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        bsz = len(sample['hypos'])
        nhypos = len(sample['hypos'][0])

        sample['hypos'] = [[h[0]] for h in sample['hypos']]
        nhypos = 1

        # get costs for hypotheses using --seq-scorer (defaults to 1. - BLEU)
        rewards = self.task.get_reward(sample, sentence_level=self.sentence_level_reward)
        rewards = rewards.detach()

        if self.args.normalize_costs:
            unnormalized_costs = rewards.clone()
            max_costs = rewards.max(dim=1, keepdim=True)[0]
            min_costs = rewards.min(dim=1, keepdim=True)[0]
            rewards = (rewards - min_costs) / (max_costs - min_costs).clamp_(min=1e-6)

            # norms = rewards.norm(p=2, dim=1, keepdim=True)
            # rewards = rewards.div(norms.expand_as(rewards))
        else:
            unnormalized_costs = None

        # generate a new sample from the given hypotheses
        new_sample = self.task.get_new_sample_for_hypotheses(sample)
        hypotheses = new_sample['target'].view(bsz, nhypos, -1, 1)
        hypolen = hypotheses.size(2)
        pad_mask = hypotheses.ne(self.task.target_dictionary.pad())
        lengths = pad_mask.sum(dim=2).float()

        net_output, _ = model(**new_sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(bsz, nhypos, hypolen, -1)

        scores = lprobs.gather(3, hypotheses)
        scores *= pad_mask.float()

        if self.sentence_level_reward:
            scores = scores.sum(dim=2) / lengths

        # if not use_enc_for_baseline:
        rewards = rewards.unsqueeze(1).unsqueeze(3) if not self.sentence_level_reward else rewards.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        # use_enc_for_baseline=True -> baseline_reward.size() = (bsz, 1)
        # use_enc_for_baseline=False -> baseline_reward.size() = (hypolen, bsz, 1)
        if self.use_baseline_reward:
            base_input = sample['net_enc_output']['encoder_out'] if use_enc_for_baseline else net_output[1]['inner_states'][-1]
            baseline_reward = self.baseline(base_input, mean=use_enc_for_baseline).float()

            if use_enc_for_baseline:
                baseline_reward = baseline_reward.unsqueeze(2).unsqueeze(3).expand_as(rewards).clone()
            else:
                baseline_reward = baseline_reward.view(bsz, 1, hypolen, -1)

            if not self.sentence_level_reward:
                baseline_reward *= pad_mask.float()

            if self.optimize_baseline_reward and not self.a2c:
                self.baseline_optimizer.zero_grad()
                bl = torch.nn.MSELoss()
                baseline_loss = bl(baseline_reward, rewards)
                baseline_loss.backward(retain_graph=True)
                self.baseline_optimizer.step()


        # eps = np.finfo(np.float32).eps.item()
        # rewards = (rewards - rewards.mean(axis=2).unsqueeze(3)) / (rewards.std(axis=2).unsqueeze(3) + eps)


        if self.use_baseline_reward:
            if not self.a2c:
                if self.num_steps < self.num_warmup_steps:
                    loss = (-rewards * scores).sum()
                else:
                    loss = (-(rewards - baseline_reward) * scores).sum()
            else:
                # a2c
                entropy = Categorical(F.softmax(net_output[0].float(), dim=-1)).entropy().mean()
                rewards = rewards - baseline_reward
                actor_loss = -(scores * rewards.detach()).mean()
                critic_loss = rewards.pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        else:
            loss = (-rewards * scores).sum()

        # self.num_steps += 1

      
        # avg_scores = scores.sum(dim=2) / lengths
        # probs = F.softmax(avg_scores, dim=1).squeeze(-1)
        # loss = (probs * costs).sum()

        sample_size = bsz
        assert bsz == utils.item(rewards.size(dim=0))
        logging_output = {
            'loss': utils.item(loss.data),
            'num_cost': rewards.numel(),
            'ntokens': sample['ntokens'],
            'nsentences': bsz,
            'sample_size': sample_size,
        }

        def add_cost_stats(costs, prefix=''):
            logging_output.update({
                prefix + 'sum_cost': utils.item(costs.sum()),
                prefix + 'min_cost': utils.item(costs.min(dim=1)[0].sum()),
                prefix + 'cost_at_1': utils.item(costs[:, 0].sum()),
            })

        add_cost_stats(rewards)
        if unnormalized_costs is not None:
            add_cost_stats(unnormalized_costs, 'unnormalized_')

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        num_costs = sum(log.get('num_cost', 0) for log in logging_outputs)
        agg_outputs = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
        }

        def add_cost_stats(prefix=''):
            agg_outputs.update({
                prefix + 'avg_cost': sum(log.get(prefix + 'sum_cost', 0) for log in logging_outputs) / num_costs,
                prefix + 'min_cost': sum(log.get(prefix + 'min_cost', 0) for log in logging_outputs) / nsentences,
                prefix + 'cost_at_1': sum(log.get(prefix + 'cost_at_1', 0) for log in logging_outputs) / nsentences,
            })

        add_cost_stats()
        if any('unnormalized_sum_cost' in log for log in logging_outputs):
            add_cost_stats('unnormalized_')

        return agg_outputs
