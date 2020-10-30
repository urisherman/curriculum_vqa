import torch
import torch.nn as nn
import torch.nn.functional as F

from cvqa.model_dev import misc
from cvqa.utils import device


class Seq2SeqLSTM(nn.Module):

    @staticmethod
    def args(input_vocab, target_vocab):
        return {
            'V_input': len(input_vocab),
            'V_target': len(target_vocab),
            'd_input': 16,
            'd_target': 16,
            'd_h': 16,
            'num_layers': 2,
            'bos_index': target_vocab.bos_index,
            'eos_index': target_vocab.eos_index,
            'pad_index': target_vocab.pad_index,
        }

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_h = args['d_h']
        self.num_layers = args['num_layers']

        self.E_input = nn.Embedding(args['V_input'], args['d_input'])
        self.encoder_lstm = nn.LSTM(args['d_input'], args['d_h'], num_layers=args['num_layers'])
        self.decoder_lstm = nn.LSTM(args['d_target'], args['d_h'], num_layers=args['num_layers'])
        self.W_o = nn.Linear(args['d_h'], args['d_target'])
        self.E_target = nn.Embedding(args['V_target'], args['d_target'])

    def init_hiddens(self, B):
        h_0 = torch.zeros(self.num_layers, B, self.d_h).to(device)
        c_0 = torch.zeros(self.num_layers, B, self.d_h).to(device)
        return h_0, c_0

    def forward(self, X, decoder_input, decoder_hidden_state=None):
        """
        X: [B, N_seq]
        target: [B, N_seq_t]

        returns logits for target tokens: [B, N_seq_t, V_target]
        """
        B, N_seq = X.shape
        if decoder_hidden_state is None:
            X_embed = self.E_input(X)
            X_embed = X_embed.transpose(0, 1)  # [N_seq, B, d_input]

            h_0, c_0 = self.init_hiddens(B)
            encoder_out, (enc_h_n, enc_c_n) = self.encoder_lstm(X_embed, (h_0, c_0))
            decoder_hidden_state = (enc_h_n, enc_c_n)

        target_embed = self.E_target(decoder_input)
        target_embed = target_embed.transpose(0, 1)
        decoder_out, decoder_hidden_state = self.decoder_lstm(target_embed, decoder_hidden_state)

        # decoder_out: [N_seq_t, B, d_h]
        decoder_out = decoder_out.transpose(0, 1)
        # Add here decoder_out.att( encoder hiddens )  --> Section 3.3 in Dong & Lapata
        pred_target_vecs = self.W_o(decoder_out)
        logits = F.linear(pred_target_vecs, self.E_target.weight)

        return logits, decoder_hidden_state

    def forward_train(self, sample: dict):
        X = sample['prompt']
        Y = sample['target_program_in']
        logits, decoder_hidden_state = self.forward(X, Y)
        return logits

    def forward_test(self, sample: dict, max_len=30):
        y_true = sample['target_program_in']
        max_len = y_true.shape[1]

        X = sample['prompt']
        B, N_seq = X.shape

        B_update_mask = torch.ones(B, dtype=torch.bool).to(device)
        y_pred = torch.ones(B, max_len, dtype=torch.long).to(device) * self.args['pad_index']
        logits = torch.zeros(B, max_len, self.args['V_target']).to(device)
        y_prev_token = torch.ones(B, 1, dtype=torch.long).to(device) * self.args['bos_index']
        decode_state = None
        for t in range(max_len):
            y_next_logit, decode_state = self(X, y_prev_token, decode_state)
            _, y_next_token = torch.max(y_next_logit, axis=-1)
            logits[B_update_mask, t:t+1, :] = y_next_logit[B_update_mask]
            y_pred[B_update_mask, t:t+1] = y_next_token[B_update_mask]
            y_prev_token = y_next_token
            B_update_mask = torch.logical_and(B_update_mask, y_next_token.flatten() != self.args['eos_index'])
        return logits, y_pred


if __name__ == '__main__':
    X_all, vocab_X, X_txt = misc.random_sentences('abcdefghijklmnop', N_samples=10, sentence_len=5, N_words=100, word_len=4)
    Y_all, vocab_Y, Y_txt = misc.random_sentences('ABCDEFGHIJKLMNOP', N_samples=10, sentence_len=5, N_words=100, word_len=4)

    from tqdm import tqdm

    model_args = Seq2SeqLSTM.args(vocab_X, vocab_Y)
    model = Seq2SeqLSTM(model_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    with tqdm(range(200)) as pbar:
        for i in pbar:
            batch_idxs = torch.randint(len(X_all), [32])
            X_batch = X_all[batch_idxs]
            Y_batch = Y_all[batch_idxs]

            optimizer.zero_grad()
            Y_logits, _ = model(X_batch, Y_batch)

            loss = loss_fn(Y_logits.flatten(end_dim=1)[:-1], Y_batch.flatten()[1:])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if i % 100 == 0:
                pbar.set_description(f'Loss={loss.item()}')

    logits, _ = model(X_all[:5], Y_all[:5])
    _, y_pred = torch.max(logits, axis=-1)
    print(y_pred[:, :-1])

    logits, y_pred = model.forward_test({'prompt': X_all[:5]})
    print(y_pred)