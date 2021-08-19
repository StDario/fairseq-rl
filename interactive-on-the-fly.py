#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import copy

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import data_utils
from fairseq import optim


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )

def build_optimizer(args, model):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # if self.args.fp16:
    #     if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
    #         print('| WARNING: your device does NOT support faster training with --fp16, '
    #               'please switch to FP32 which is likely to be faster')
    #     if self.args.memory_efficient_fp16:
    #         self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
    #     else:
    optimizer = optim.FP16Optimizer.build_optimizer(args, params)
    # else:
    #     if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
    #         print('| NOTICE: your device may support faster training with --fp16')
    #     self._optimizer = optim.build_optimizer(self.args, params)

    # if self.args.use_bmuf:
    #     self._optimizer = optim.FairseqBMUF(self.args, params, self._optimizer)

    # We should initialize the learning rate scheduler immediately after
    # building the optimizer, so that the initial learning rate is set.
    # self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
    # self._lr_scheduler.step_update(0)

    return optimizer

def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1


    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    def merge(samples, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s for s in samples],
            pad_idx, eos_idx, False, move_eos_to_beginning,
        )

    def compute_loss(model, net_output, sample, reduce=True):
        eps = 0.1
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(pad_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = eps / lprobs.size(-1)
        loss = (1. - eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss


    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    # print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        # model.make_generation_fast_(
        #     beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        #     need_attn=args.print_alignment,
        # )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    args.optimizer = 'adam'
    args.lr = [0.0002]
    args.adam_betas = '(0.9, 0.98)'
    args.adam_eps = 1e-08
    args.update_freq = [1]
    args.distributed_world_size = 1
    num_iterations = 7


    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    pad_idx = src_dict.pad()
    eos_idx = src_dict.eos()



    # Initialize generator
    generator = task.build_generator(args)

    # Hack to support GPT-2 BPE
    if args.remove_bpe == 'gpt2':
        from fairseq.gpt2_bpe.gpt2_encoding import get_encoder
        decoder = get_encoder(
            'fairseq/gpt2_bpe/encoder.json',
            'fairseq/gpt2_bpe/vocab.bpe',
        )
        encode_fn = lambda x: ' '.join(map(str, decoder.encode(x)))
    else:
        decoder = None
        encode_fn = lambda x: x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    # if args.buffer_size > 1:
    #     print('| Sentence buffer size:', args.buffer_size)
    # print('| Type the input sentence and press return:')

    batches = []

    f_ft_src = open(args.ft_parallel + '.src')
    f_ft_trg = open(args.ft_parallel + '.trg')
    id = 0

    for line_src, line_trg in zip(f_ft_src, f_ft_trg):

        if len(line_src.strip()) == 0:
            batches.append(None)
            continue

        contents_src = line_src.split('\t')
        contents_trg = line_trg.split('\t')

        src_toks = []
        trg_toks = []

        for c_src, c_trg in zip(contents_src, contents_trg):

            src_tok = src_dict.encode_line(
                c_src.strip(), add_if_not_exist=False,
                append_eos=True, reverse_order=False,
            ).long()

            trg_tok = tgt_dict.encode_line(
                c_trg.strip(), add_if_not_exist=False,
                append_eos=True, reverse_order=False,
            ).long()

            src_toks.append(src_tok)
            trg_toks.append(trg_tok)



        src_tokens = merge(src_toks)
        src_lengths = src_tokens.new([s.numel() for s in src_toks])
        # id = src_tokens.new([s['id'] for s in samples])

        target = merge(trg_toks)
        ntokens = sum(len(t) for t in trg_toks)

        prev_output_tokens = merge(
            trg_toks,
            move_eos_to_beginning=True,
        )

        batch = {
            'id': id,
            'nsentences': len(src_tokens),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'prev_output_tokens': prev_output_tokens
            },
            'target': target,
            'trg_tokens': trg_toks,

        }

        id += 1

        batches.append(batch)
        
    f_ft_src.close()
    f_ft_trg.close()

    ind = 0
    ft_batch_size = 5

    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):

        batch = batches[ind]
        ind += 1

        if batch is not None:
            start_ind = 0
            
            model_copy = copy.deepcopy(models[0])

            optimizer = build_optimizer(args, model_copy)

            model_copy.train()

            indices = []
            sent_count = 0
            e_ind = 0
            n_sents = batch['nsentences']
            while True:
                s_ind = e_ind
                e_ind = e_ind
                max_word_count = 0

                while True:
                    # sent_count += len(batch['trg_tokens'][e_ind])
                    max_word_count = max(max_word_count, batch['net_input']['src_lengths'][e_ind], len(batch['trg_tokens'][e_ind]))
                    
                    if max_word_count * (e_ind - s_ind) >= 2048:
                        break

                    e_ind += 1

                    if e_ind == n_sents:
                        break

                indices.append((s_ind, e_ind))

                if e_ind == n_sents:
                    break

            
            for inds in indices:

                start_ind = inds[0]
                end_ind = inds[1]

                train_batch = {}
                train_batch['id'] = batch['id']
                train_batch['ntokens'] = sum(len(t) for t in batch['trg_tokens'][start_ind:end_ind])
                train_batch['target'] = batch['target'][start_ind:end_ind]
                train_batch['trg_tokens'] = batch['trg_tokens'][start_ind:end_ind]
                train_batch['net_input'] = {}
                train_batch['net_input']['src_tokens'] = batch['net_input']['src_tokens'][
                                                         start_ind:end_ind]
                train_batch['net_input']['src_lengths'] = batch['net_input']['src_lengths'][
                                                          start_ind:end_ind]
                train_batch['net_input']['prev_output_tokens'] = batch['net_input']['prev_output_tokens'][
                                                                 start_ind:end_ind]

             
                max_len = max(train_batch['net_input']['src_lengths'])
                # new_src_tokens = train_batch['net_input']['src_tokens'].new(train_batch['net_input']['src_tokens'].size(0), max_len + 1)
                new_src_tokens = train_batch['net_input']['src_tokens'][:,:max_len + 1]
                train_batch['net_input']['src_tokens'] = new_src_tokens

                max_len = max([t.numel() for t in train_batch['trg_tokens']])
                # new_tr_tokens = train_batch['target'].new(
                #     train_batch['target'].size(0), max_len + 1)
                new_tr_tokens = train_batch['target'][:, :max_len + 1]
                train_batch['target'] = new_tr_tokens

                # new_prev_output_tokens = train_batch['net_input']['prev_output_tokens'].new(
                #     train_batch['net_input']['prev_output_tokens'].size(0), max_len + 1)
                new_prev_output_tokens = train_batch['net_input']['prev_output_tokens'][:, :max_len + 1]
                train_batch['net_input']['prev_output_tokens'] = new_prev_output_tokens

                
                
                is_ok = False
                while not is_ok:
                    try:
                        train_batch = utils.move_to_cuda(train_batch)
                        net_output, _ = model_copy(**train_batch['net_input'])
                        loss, nll_loss = compute_loss(model_copy, net_output, train_batch, reduce=True)

                        optimizer.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
                        is_ok = True

                        del train_batch
                        torch.cuda.empty_cache()
                    except:
                        utils.move_to_cpu(train_batch)
                        torch.cuda.empty_cache()

                        half_index = int(len(train_batch['trg_tokens']) / 2)
                        train_batch1 = {}
                        train_batch1['id'] = train_batch['id']
                        train_batch1['ntokens'] = sum(len(t) for t in train_batch['trg_tokens'][:half_index])
                        train_batch1['target'] = train_batch['target'][:half_index]
                        # train_batch1['trg_tokens'] = train_batch['trg_tokens'][:half_index]
                        train_batch1['net_input'] = {}
                        train_batch1['net_input']['src_tokens'] = train_batch['net_input']['src_tokens'][
                                                                 :half_index]
                        train_batch1['net_input']['src_lengths'] = train_batch['net_input']['src_lengths'][
                                                                  :half_index]
                        train_batch1['net_input']['prev_output_tokens'] = train_batch['net_input']['prev_output_tokens'][
                                                                         :half_index]

                        train_batch1 = utils.move_to_cuda(train_batch1)
                        net_output, _ = model_copy(**train_batch1['net_input'])
                        loss, nll_loss = compute_loss(model_copy, net_output, train_batch1, reduce=True)

                        optimizer.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()

                        del train_batch1
                        torch.cuda.empty_cache()

                        train_batch1 = {}
                        train_batch1['id'] = train_batch['id']
                        train_batch1['ntokens'] = sum(len(t) for t in train_batch['trg_tokens'][half_index:])
                        train_batch1['target'] = train_batch['target'][half_index:]
                        # train_batch1['trg_tokens'] = train_batch['trg_tokens'][half_index:]
                        train_batch1['net_input'] = {}
                        train_batch1['net_input']['src_tokens'] = train_batch['net_input']['src_tokens'][
                                                                  half_index:]
                        train_batch1['net_input']['src_lengths'] = train_batch['net_input']['src_lengths'][
                                                                   half_index:]
                        train_batch1['net_input']['prev_output_tokens'] = train_batch['net_input']['prev_output_tokens'][
                                                                          half_index:]

                        train_batch1 = utils.move_to_cuda(train_batch1)
                        net_output, _ = model_copy(**train_batch1['net_input'])
                        loss, nll_loss = compute_loss(model_copy, net_output, train_batch1, reduce=True)

                        optimizer.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()

                        del train_batch1
                        torch.cuda.empty_cache()

                        is_ok = True

                    if is_ok:
                        break

            model_copy.eval()

            model_copy.make_generation_fast_(
                beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                need_attn=args.print_alignment,
            )
            copy_models = [model_copy]
        else:
            copy_models = [models[0]]


        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = task.inference_step(generator, copy_models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

       
        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
               
            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                if decoder is not None:
                    hypo_str = decoder.decode(map(int, hypo_str.strip().split()))
                print(hypo_str)
                
        # update running id counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
