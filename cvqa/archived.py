# def fairseq_train_step_walkthrough(sample_batch, model, optimizer, criterion):
#     # to make reproducible with checkpoints, set seed here appropriately (see example in LabelSmoothedCrossEntropyCriterion)
#
#     model.train()
#     #     criterion.train()
#     optimizer.zero_grad()
#
#     # trainer._prepare_sample():
#     #   move sample to cuda
#     #   fp16
#
#     #     sample_batch = {
#     #         'id': Tensor[B],
#     #         'nsentences': B,
#     #         'ntokens': ?,
#     #         'net_input': {src_tokens, src_lengths, prev_output_tokens},
#     #         'target': Tensor[B, No]
#     #     }
#
#     #     LabelSmoothedCrossEntropyCriterion.forward():
#     #         model_output = TransformerModel.forward(
#     #           src_tokens: Tensor[B, Ni], # the token ids
#     #           src_lengths:Tensor[B], #  individual lengths of each sample (I guess they are padded, not sure why this is needed)
#     #           prev_output_tokens: Tensor[B, No] # for teacher forcing!
#     #         )
#     model_output = model(**sample_batch['net_input'])
#     #     model_output: (
#     #         0: Tensor[B, No, V], --> real output, aka logits, the unnormalized scores over each token
#     #         1: {
#     #             attn: Tensor: [B, No, Ni],  --> cross attention
#     #             inner_states: list[Tensor: [?, ?, d]]  --> looks like decoder internal layers
#     #         }
#     #     )
#
#     # logits --softmax--> probabilities --log--> log probabilities
#     ##   lprobs = model.get_normalized_probs(y_hat, log_probs=True)
#     ##   also handles prepare_for_onnx_export_
#
#     lprobs = F.log_softmax(x, dim=-1, dtype=torch.float32)  # [B, No, V]
#     lprobs = lprobs.view(-1, lprobs.size(-1))  # [B*No, V]
#     targets = sample_batch['target'].view(-1, 1)  # [B*No]
#
#     # loss, nll_loss = fairseq_utils.label_smoothed_nll_loss(
#     #     lprobs, target, args.label_smoothing, ignore_index=vocab.pad_index,
#     # )
#     sample_size = sample_batch['target'].size(0) if args.sentence_avg else sample_batch['ntokens']
#     #     logging_output = {
#     #             'loss': utils.item(loss.data) if reduce else loss.data,
#     #             'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
#     #             'ntokens': sample['ntokens'],
#     #             'nsentences': sample['target'].size(0),
#     #             'sample_size': sample_size,
#     #         }
#
#     optimizer.backward(loss)
#     optimizer.multiply_grads(1. / float(sample_size))  # verify
#     #     grad_norm = optimizer.clip_grad_norm(self.args.clip_norm)
#
#     optimizer.step()
#     #     self.set_num_updates(self.get_num_updates() + 1)
#     fairseq_utils.clear_cuda(args, 0)
#     return logging_output





# EINSUM tests

# B = 2
# d = 3
# c = 4
# P = 5
#
# op = np.random.randn(P, d, c)
# p_enc = np.random.randn(B, d)
# img_feat = np.random.randn(B, P)
#
# p_ops = np.einsum('pdc,bd->bpc', op, p_enc)
#
# np.einsum('bpc,bp->bc', p_ops, img_feat)
# np.einsum('bp,bpc->bc', img_feat, p_ops)
# np.einsum('bpc,bp->bc', p_ops[[0]], img_feat[[0]])
# torch.einsum('pdc,bd->bpc', viz_model.W_op, prompt_encoded)