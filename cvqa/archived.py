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



#### upload to google and unzip from stream ###
# from google.colab import files
# uploaded = files.upload()
# import shutil
# import zipfile
# from io import BytesIO
#
# zipdata = BytesIO(uploaded[f'{curriculum_name}.zip'])
#
# with zipfile.ZipFile(zipdata, 'r') as zip_ref:
#     zip_ref.extractall(curriculum_root)


# class NoamOpt:
#     "Optim wrapper that implements rate."
#
#     def __init__(self, model_size, factor, warmup, optimizer):
#         self.optimizer = optimizer
#         self._step = 0
#         self.warmup = warmup
#         self.factor = factor
#         self.model_size = model_size
#         self._rate = 0
#
#     def step(self):
#         "Update parameters and rate"
#         self._step += 1
#         rate = self.rate()
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self._rate = rate
#         self.optimizer.step()
#
#     def zero_grad(self):
#         self.optimizer.zero_grad()
#
#     def rate(self, step=None):
#         "Implement `lrate` above"
#         if step is None:
#             step = self._step
#         return self.factor * \
#                (self.model_size ** (-0.5) *
#                 min(step ** (-0.5), step * self.warmup ** (-1.5)))
#
#
# class UriOpt:
#     "Optim wrapper that implements rate."
#
#     def __init__(self, decay_steps_interval, factor, optimizer, clif=100):
#         self.optimizer = optimizer
#         self._step = 0
#         self.factor = factor
#         self.decay_steps_interval = decay_steps_interval
#         self._rate = 0
#         self.lr = next(iter(optimizer.param_groups))['lr']
#         self.clif = clif
#
#     def step(self):
#         "Update parameters and rate"
#         self._step += 1
#         rate = self.rate()
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self._rate = rate
#         self.optimizer.step()
#
#     def zero_grad(self):
#         self.optimizer.zero_grad()
#
#     def rate(self, step=None):
#         "Implement `lrate` above"
#         if step is None:
#             step = self._step
#         if step < self.clif:
#             return self.lr
#         else:
#             factor_steps = (step - self.clif) / self.decay_steps_interval
#             return self.lr * self.factor ** (-factor_steps)

# import matplotlib.pyplot as plt
#
# optimizer = UriOpt(63, 2,
#             torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9))
#
# lrs = []
# for i in range(600):
#   lrs.append(optimizer.rate(i+1))
#
# plt.plot(lrs);
# lrs[-1]




###
### Fairseq model running in colab
###
# seed = 1
# np.random.seed(seed)
# torch.manual_seed(seed)
#
# vocab = train_dataset.vocab
#
# struct_viz = True
#
# d=16
# img_output_features=3
#
# tokens_embed = fairseq_misc.build_embedding(vocab, d)
# encoder = fairseq_misc.build_vqa_encoder(
#     vocab, tokens_embed,
#     ffn_dim=d*2,
#     layers=2,
#     attention_heads = 2
# )
# decoder = fairseq_misc.build_decoder(
#     vocab, tokens_embed,
#     ffn_dim=d*2,
#     layers=2,
#     attention_heads=2
# )
#
# if struct_viz:
#   img_perceptor = models.StructuredImageModel(
#       train_dataset.struct_viz_vocab, d, img_output_features)
# else:
#   img_perceptor = models.BasicImgModel(d, img_output_features)
#
# train_dataset.use_viz_rep = struct_viz
# dev_dataset.use_viz_rep = struct_viz
#
# model = models.VQAModelV1(encoder, img_perceptor, decoder, bos=vocab.bos_index)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
# # optimizer = UriOpt(1200, 150, 1.3, optimizer)
#
# trainer = trainers.VQATrainer(log_dir='runs/basic')
# train_loss, train_acc, dev_acc = trainer.train(
#     model, train_dataset, dev_dataset,
#     optimizer, num_epochs=100, batch_size=32
# )





# self.prompts = [
#             ('shape', 'Which shape is this?', '[shape]'),
#             ('shape', 'This item is a...?', '[shape]'),
#             ('shape', 'The item in the image is a...?', '[shape]'),
#             ('shape', 'The shape of the item in the image is a...?', '[shape]'),
#             ('shape_yes_no', 'Is the item in the image a [shape]?', 'Yes'),
#             ('shape_yes_no', 'This is a [shape].', 'True'),
#             ('shape_yes_no', 'This is not a [shape].', 'Wrong'),
#             ('color', 'What is the color of this item?', '[color]'),
#             ('color', 'What color is this item?', '[color]'),
#             ('color', 'The color of this item is...?', '[color]'),
#             ('color', 'The color of the item in the image is...?', '[color]'),
#             ('color_yes_no', 'Is the color of the item in the image [color]?', 'Yes'),
#             ('color_yes_no', 'This is a [color] item.', 'True'),
#             ('color_yes_no', 'This is a not a [color] item.', 'Wrong'),
#         ]

#
# if 'viz_rep' in sample:
#     viz_rep = sample['viz_rep']
#     # viz_rep['encoded'] = encode_line(viz_rep['shape'] + ' ' + viz_rep['color'], struct_viz_vocab)
#     for o in viz_rep['objects']:
#         o['encoded_tokens'] = encode_line(o['tokens'], struct_viz_vocab)
#
# new_samples.append(sample)
#
# ### Make structured image representation proper
# if 'viz_rep' in new_samples[0]:
#     N_objs_max = max(map(lambda s: len(s['viz_rep']['objects']), new_samples))
#     for i, sample in enumerate(new_samples):
#         viz_rep = sample['viz_rep']
#
#         obj_tokens = list(map(lambda ob: ob['encoded_tokens'], viz_rep['objects']))
#         toekns_img = torch.stack(obj_tokens)  # [N_objs, N_tokens]
#
#         obj_numerics = list(map(lambda ob: ob['numerics'], viz_rep['objects']))
#         numerics_img = torch.stack(obj_numerics)  # [N_objs, N_numerics]
#
#         pad_len = N_objs_max - toekns_img.shape[0]
#
#         toekns_img = F.pad(toekns_img, (0, 0, 0, pad_len), value=struct_viz_vocab.pad_index)
#         numerics_img = F.pad(numerics_img, (0, 0, 0, pad_len), value=struct_viz_vocab.pad_index)
#
#         mask = torch.ones(N_objs_max)
#         mask[toekns_img.shape[0]:] = 0
#         viz_rep['tokens_img'] = toekns_img
#         viz_rep['numerics_img'] = numerics_img
#         viz_rep['mask_img'] = mask


# xw_logits_init = torch.ones(B, N_objs).to(device)*-.01
# xw_logits = self.m_f1(X, xw_logits_init, f1_op)
# neg_logits = torch.log(1 - torch.exp(xw_logits))
# full_logits = torch.stack([neg_logits, xw_logits], dim=-1)
# return full_logits