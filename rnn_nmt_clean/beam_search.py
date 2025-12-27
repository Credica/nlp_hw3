"""
Beam Search Decoder for RNN-based Neural Machine Translation
WARNING: This implementation has SEVERE bugs - see BEAM_SEARCH_FIX_REPORT.md

Key Issues:
1. All beams share the SAME hidden state (should be separate)
2. All beams use the SAME input token (only beam 0's token)
3. All beams use the SAME probability distribution (expanded)
4. This effectively does GREEDY decoding, not beam search
5. Different beam sizes produce IDENTICAL results
"""
import torch
import torch.nn.functional as F
import config_v2 as config

class Beam:
    """Beam search for one sentence - BUGGY VERSION"""

    def __init__(self, size, pad_idx, bos_idx, eos_idx, device):
        self.size = size
        self.device = device
        self._done = False

        # The score for each translation on the beam
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step
        self.prev_ks = []

        # The outputs at each time-step
        self.next_ys = [torch.full((size,), pad_idx, dtype=torch.long, device=device)]
        self.next_ys[0][0] = bos_idx

    def get_current_state(self):
        """Get the outputs for the current timestep"""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current timestep"""
        return self.prev_ks[-1] if len(self.prev_ks) > 0 else None

    @property
    def done(self):
        return self._done

    def advance(self, word_logprob):
        """Update beam status and check if finished"""
        num_words = word_logprob.size(1)

        # Sum the previous scores
        if len(self.prev_ks) > 0:
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            # In the first step, all beams start with BOS token
            beam_lk = word_logprob[0]

        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS
        if self.next_ys[-1][0].item() == config.EOS_TOKEN:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        """Sort the scores"""
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """Get the score of the best in the beam"""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]  # NOTE: Returns 2nd best, not 1st!

    def get_tentative_hypothesis(self):
        """Get the decoded sequence for the current timestep"""
        if len(self.next_ys) == 1:
            dec_seq = list(self.next_ys[0].cpu().numpy())
            return dec_seq

        dec_seq = []
        j = 0
        for j in range(len(self.next_ys) - 1, -1, -1):
            if self.next_ys[j][0].item() != config.PAD_TOKEN:
                break

        curr_k = 0
        for j in range(len(self.next_ys) - 1, -1, -1):
            dec_seq.append(self.next_ys[j][curr_k].item())  # NOTE: Always uses curr_k=0!
            if self.next_ys[j][curr_k].item() == config.SOS_TOKEN:
                break

        return dec_seq[::-1]


def beam_search_decode(model, src, max_len=100, beam_size=5):
    """Beam search decoding for one sentence - BUGGY VERSION"""
    model.eval()

    with torch.no_grad():
        # Encode
        src = src.unsqueeze(1).to(config.device)  # Add batch dimension
        encoder_outputs, hidden = model.encoder(src)
        # hidden shape: [n_layers, 1, hidden_dim]

        # Initialize beam
        beam = Beam(
            size=beam_size,
            pad_idx=0,
            bos_idx=2,
            eos_idx=3,
            device=config.device
        )

        # Decode
        for step in range(max_len):
            if beam.done:
                break

            # For each beam, we need to decode with its own hidden state
            # But RNN hidden state is shared, so we use the SAME hidden state for all beams
            # and rely on the input sequence to differentiate them

            # Get current input token
            if step == 0:
                input_token = 2  # BOS token
            else:
                # Get the token chosen by the best beam
                best_beam_id = beam.prev_ks[-1][0]  # NOTE: Uses prev_ks incorrectly!
                input_token = beam.next_ys[-1][best_beam_id].item()

            # Decode one step
            input_tensor = torch.tensor([input_token]).to(config.device)
            output, new_hidden, _ = model.decoder(input_tensor, hidden, encoder_outputs)

            # Get log probabilities
            log_probs = F.log_softmax(output, dim=1)
            # log_probs shape: [1, vocab_size]

            # Expand to all beams (all beams get the same probabilities at this step)
            # This is correct for RNN because they share the same hidden state
            # The beams will diverge based on their different paths in subsequent steps
            log_probs_expanded = log_probs.expand(beam.size, -1)
            beam.advance(log_probs_expanded)

            # Update hidden state for next iteration
            hidden = new_hidden

    # Get best hypothesis
    hypothesis = beam.get_tentative_hypothesis()

    # Remove EOS and SOS tokens
    hypothesis = [token for token in hypothesis if token not in [2, 3]]

    return hypothesis


def greedy_decode(model, src, max_len=100):
    """Greedy decoding for one sentence"""
    model.eval()

    with torch.no_grad():
        src = src.unsqueeze(1).to(config.device)

        # Encode
        encoder_outputs, hidden = model.encoder(src)

        # Decode
        input_token = 2  # BOS token
        translation_ids = [2]

        for _ in range(max_len):
            input_tensor = torch.tensor([input_token]).to(config.device)
            output, hidden, _ = model.decoder(input_tensor, hidden, encoder_outputs)

            predicted_token = output.argmax(1).item()
            translation_ids.append(predicted_token)

            if predicted_token == 3:  # EOS token
                break

            input_token = predicted_token

    # Remove EOS and SOS tokens
    translation_ids = [token for token in translation_ids if token not in [2, 3]]

    return translation_ids
