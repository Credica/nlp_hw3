"""
Correct Beam Search Implementation - Following Google's Official Algorithm

This implementation follows the standard beam search algorithm used in TensorFlow's
official translation models. The key insight is that at each step, we consider
ALL possible continuations from ALL current beams, then select the top-k.

Algorithm:
1. Start with beam_width initial hypotheses (all starting with BOS)
2. At each time step:
   - For EACH current beam, compute logits for all vocabulary
   - Combine all logits with previous beam scores
   - Apply length penalty and masking
   - Select top-k hypotheses from ALL possibilities
3. Track beam parents to reconstruct paths
"""

import torch
import torch.nn.functional as F
import config_v2 as config


def beam_search_decode(model, src, beam_size=5, max_len=100, length_penalty=0.6):
    """
    Beam search decoding - CORRECT IMPLEMENTATION

    Args:
        model: The seq2seq model
        src: Source sequence [src_len]
        beam_size: Number of beams to keep
        max_len: Maximum decoding length
        length_penalty: Alpha parameter for length penalty (0.6-0.7 works well)

    Returns:
        List of token IDs for the best translation (without special tokens)
    """
    model.eval()

    with torch.no_grad():
        # Encode the source sentence
        src = src.unsqueeze(1).to(config.device)  # [src_len, 1]

        # Get encoder outputs and hidden state
        encoder_outputs, hidden = model.encoder(src)

        # Special token IDs (hardcoded based on the tokenizer setup)
        PAD_ID = 0
        SOS_ID = 2
        EOS_ID = 3

        # Initialize beams
        # Each beam contains: tokens, log_prob, length, hidden state, finished flag
        # beams = [{
        #     'tokens': [SOS_ID],
        #     'log_prob': 0.0,
        #     'length': 1,
        #     'hidden': hidden.clone(),
        #     'finished': False
        # } for _ in range(beam_size)]

        beams = [{
            'tokens': [SOS_ID],
            'log_prob': 0.0,
            'length': 1,
            'hidden': hidden.clone(),
            'finished': False
        } ]

        # Decode step by step
        for step in range(max_len):
            # Collect all candidate hypotheses from all beams
            all_candidates = []

            # For each current beam, expand it
            for beam_id, beam in enumerate(beams):
                # Skip if this beam is already finished
                # if beam['finished']:
                #     # For finished beams, we only propagate EOS token
                #     candidate = {
                #         'tokens': beam['tokens'] + [EOS_ID],
                #         'log_prob': beam['log_prob'],
                #         'length': beam['length'],
                #         'hidden': beam['hidden'],
                #         'finished': True,
                #         'parent_id': beam_id
                #     }
                #     all_candidates.append(candidate)
                #     continue
                if beam['finished']:
                    all_candidates.append(candidate)
                    continue
                # Get the last token from this beam
                input_token = torch.tensor([beam['tokens'][-1]]).to(config.device)

                # Decode one step
                output, new_hidden, _ = model.decoder(
                    input_token,
                    beam['hidden'],
                    encoder_outputs
                )

                # Get log probabilities for all vocabulary items
                log_probs = F.log_softmax(output.squeeze(0), dim=0)
                # log_probs shape: [vocab_size]

                # Get top-k candidates from this beam
                # We take top_k = beam_size to ensure we have enough candidates
                top_log_probs, top_tokens = log_probs.topk(beam_size)

                # Create candidates for each top token
                for log_prob, token in zip(top_log_probs, top_tokens):
                    token_id = token.item()

                    # Check if this token finishes the sequence
                    is_finished = (token_id == EOS_ID)

                    # Calculate new log probability (add to existing)
                    new_log_prob = beam['log_prob'] + log_prob.item()
                    new_length = beam['length'] + 1

                    candidate = {
                        'tokens': beam['tokens'] + [token_id],
                        'log_prob': new_log_prob,
                        'length': new_length,
                        'hidden': new_hidden.clone(),
                        'finished': is_finished,
                        'parent_id': beam_id
                    }
                    all_candidates.append(candidate)

            # If no candidates, break
            if not all_candidates:
                break

            # Apply length penalty and compute scores
            # Formula: score = log_prob / (length^alpha)
            for candidate in all_candidates:
                if candidate['length'] > 1:  # Don't penalize initial token
                    candidate['score'] = candidate['log_prob'] / (
                        candidate['length'] ** length_penalty)
                else:
                    candidate['score'] = candidate['log_prob']

            # Sort all candidates by score (descending)
            all_candidates.sort(key=lambda x: x['score'], reverse=True)

            # Select top-k beams for next iteration
            beams = all_candidates[:beam_size]

            # Early termination: if all top beams are finished, we can stop
            if all(beam['finished'] for beam in beams[:min(3, len(beams))]):
                # But don't stop immediately, give it a few more steps
                if step > 10:  # Minimum decode length
                    break

        # Get the best hypothesis
        best_beam = beams[0]

        # Remove special tokens (SOS, EOS, PAD)
        tokens = [t for t in best_beam['tokens']
                 if t not in [SOS_ID, EOS_ID, PAD_ID]]

        return tokens


def batch_beam_search_decode(model, batch_src, beam_size=5,
                             max_len=100, length_penalty=0.6):
    """
    Batch beam search decoding

    Args:
        model: The seq2seq model
        batch_src: Batch of source sequences [src_len, batch_size]
        beam_size: Number of beams to keep
        max_len: Maximum decoding length
        length_penalty: Alpha parameter for length penalty

    Returns:
        List of translations (each is a list of token IDs)
    """
    model.eval()

    with torch.no_grad():
        batch_size = batch_src.shape[1]
        translations = []

        for i in range(batch_size):
            src = batch_src[:, i]
            translation = beam_search_decode(
                model, src, beam_size=beam_size,
                max_len=max_len, length_penalty=length_penalty
            )
            translations.append(translation)

        return translations


def greedy_decode(model, src, max_len=100):
    """
    Greedy decoding (beam_size=1)
    """
    model.eval()

    with torch.no_grad():
        src = src.unsqueeze(1).to(config.device)

        # Encode
        encoder_outputs, hidden = model.encoder(src)

        # Special token IDs
        PAD_ID = 0
        SOS_ID = 2
        EOS_ID = 3

        # Decode
        tokens = [SOS_ID]
        input_token = SOS_ID

        for _ in range(max_len):
            input_tensor = torch.tensor([input_token]).to(config.device)
            output, hidden, _ = model.decoder(input_tensor, hidden, encoder_outputs)

            # Get most likely token
            log_probs = F.log_softmax(output.squeeze(0), dim=0)
            next_token = log_probs.argmax().item()

            tokens.append(next_token)

            if next_token == EOS_ID:
                break

            input_token = next_token

        # Remove special tokens
        tokens = [t for t in tokens
                 if t not in [SOS_ID, EOS_ID, PAD_ID]]

        return tokens


# Test function to verify different beam sizes produce different results
def test_beam_sizes(model, src, beam_sizes=[1, 3, 5, 10]):
    """Test that different beam sizes produce different results"""
    print("\n=== Testing Different Beam Sizes ===")
    translations = {}

    for beam_size in beam_sizes:
        translation = beam_search_decode(
            model, src, beam_size=beam_size, max_len=50
        )
        translations[beam_size] = translation
        print(f"\nBeam Size {beam_size}:")
        print(f"  Translation: {translation}")
        print(f"  Length: {len(translation)}")

    # Check if beam sizes produce different results
    if len(set(tuple(translations[b]) for b in beam_sizes)) > 1:
        print("\n✓ SUCCESS: Different beam sizes produce different results!")
    else:
        print("\n✗ FAILURE: All beam sizes produce the same result!")
        print("  This indicates a bug in the beam search implementation.")

    return translations
