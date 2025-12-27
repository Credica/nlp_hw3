"""
Comprehensive Experiment Script for RNN NMT
Compares different configurations:
1. Attention mechanisms: dot, multiplicative, additive
2. Training strategies: Teacher Forcing (100%, 50%) vs Free Running (0%)
3. Decoding strategies: Greedy vs Beam Search
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
from tqdm import tqdm
import sacrebleu
import sentencepiece as spm
import pickle

from data_loader import get_data_loaders, get_test_loader
from model_v2 import build_model, count_parameters, initialize_weights
from beam_search_correct import greedy_decode, beam_search_decode
import config_v2 as config

def train_epoch(model, train_loader, optimizer, criterion, clip, teacher_forcing_ratio):
    """Train for one epoch with specified teacher forcing ratio"""
    model.train()
    epoch_loss = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        src = batch['src'].to(config.device)
        trg = batch['trg'].to(config.device)

        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        # output shape: [trg_len, batch_size, output_dim]
        # trg shape: [trg_len, batch_size]

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return epoch_loss / len(train_loader)


def evaluate(model, valid_loader, criterion, teacher_forcing_ratio=0):
    """Evaluate on validation set"""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in valid_loader:
            src = batch['src'].to(config.device)
            trg = batch['trg'].to(config.device)

            output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(valid_loader)


def calculate_bleu(model, test_loader, zh_idx2word, en_sp, decoding_strategy='greedy', beam_size=5):
    """Calculate BLEU score on test set"""
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Calculating BLEU ({decoding_strategy})"):
            src = batch['src'].to(config.device)
            trg = batch['trg'].to(config.device)

            # Translate each sentence in batch
            for i in range(src.shape[1]):
                src_sentence = src[:, i]
                trg_sentence = trg[:, i]

                # Translate using specified strategy
                if decoding_strategy == 'greedy':
                    translation_ids = greedy_decode(model, src_sentence)
                else:  # beam search
                    translation_ids = beam_search_decode(model, src_sentence, beam_size=beam_size)

                # # Decode to text
                # translation_text = en_sp.decode_ids(translation_ids)
                # translation_text = translation_text.replace('<sos>', '').replace('<eos>', '').strip()

                # # Get reference (remove special tokens)
                # trg_text = en_sp.decode_ids(trg_sentence.tolist())
                # trg_text = trg_text.replace('<sos>', '').replace('<eos>', '').strip()
                # --- helper: strip PAD/BOS/EOS at id level ---
                PAD_ID = 0
                BOS_ID = en_sp.bos_id()  
                EOS_ID = en_sp.eos_id()  

                def strip_special(ids):
                    return [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]

                # hypothesis: greedy/beam 已经在 decode 时去掉了 special，但这里再做一次
                translation_text = en_sp.decode_ids(strip_special(translation_ids)).strip()

                # reference: 一定要 strip（trg_sentence 里含 PAD/BOS/EOS）
                trg_ids = strip_special(trg_sentence.tolist())
                trg_text = en_sp.decode_ids(trg_ids).strip()

                references.append([trg_text])
                hypotheses.append(translation_text)

    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    return bleu.score


def train_and_evaluate(attention_type, teacher_forcing_ratio, model_path):
    """Train model with specific configuration and return results"""
    print("\n" + "=" * 80)
    print(f"Experiment: Attention={attention_type}, Teacher Forcing={teacher_forcing_ratio*100:.0f}%")
    print("=" * 80)

    # Load data
    train_loader, valid_loader = get_data_loaders(config)

    # Load tokenizers for evaluation
    with open(f'{config.TOKENIZER_DIR}/zh_vocab.pkl', 'rb') as f:
        zh_data = pickle.load(f)
        zh_idx2word = zh_data['idx2word']

    en_sp = spm.SentencePieceProcessor()
    en_sp.load(f'{config.TOKENIZER_DIR}/eng.model')

    # Build model
    model = build_model(attention_type=attention_type)

    # Initialize weights
    model.apply(initialize_weights)

    # Count parameters
    n_params = count_parameters(model)
    print(f"Number of trainable parameters: {n_params:,}")

    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Training loop
    best_valid_loss = float('inf')
    patience_counter = 0
    train_losses = []
    valid_losses = []

    for epoch in range(config.N_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.CLIP, teacher_forcing_ratio)
        train_losses.append(train_loss)

        # Evaluate
        valid_loss = evaluate(model, valid_loader, criterion, teacher_forcing_ratio=0)
        valid_losses.append(valid_loss)

        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) % 60)

        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"✓ New best model saved! (valid loss: {valid_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠ No improvement for {patience_counter}/{config.EARLY_STOPPING_PATIENCE} epochs")

        # Print epoch results
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):.4f}')
        print(f'\tVal Loss: {valid_loss:.4f} | Val PPL: {math.exp(valid_loss):.4f}')

        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered!")
            print(f"Best validation loss: {best_valid_loss:.4f}")
            break

    # Load best model for evaluation
    model.load_state_dict(torch.load(model_path))

    # Calculate BLEU scores with different decoding strategies
    test_loader = get_test_loader(config)

    print("\n📊 Evaluating with different decoding strategies...")
    results = {
        'attention_type': attention_type,
        'teacher_forcing_ratio': teacher_forcing_ratio,
        'teacher_forcing': teacher_forcing_ratio,  # Add for visualization compatibility
        'best_valid_loss': best_valid_loss,
        'val_loss': best_valid_loss,  # Add for visualization compatibility
        'n_params': n_params,
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }

    # Greedy decoding
    print("\n  → Greedy decoding...")
    bleu_greedy = calculate_bleu(model, test_loader, zh_idx2word, en_sp, decoding_strategy='greedy')
    results['bleu_greedy'] = bleu_greedy
    print(f"  BLEU (Greedy): {bleu_greedy:.2f}")

    # Beam search (different beam sizes)
    best_bleu_beam = 0
    for beam_size in config.BEAM_SIZES[1:]:  # Skip beam size 1 (already did greedy)
        print(f"\n  → Beam search (size={beam_size})...")
        bleu_beam = calculate_bleu(model, test_loader, zh_idx2word, en_sp,
                                   decoding_strategy='beam', beam_size=beam_size)
        results[f'bleu_beam_{beam_size}'] = bleu_beam
        if beam_size == 3:  # Store beam-3 as default 'bleu_beam' for visualization
            results['bleu_beam'] = bleu_beam
            best_bleu_beam = bleu_beam
        print(f"  BLEU (Beam-{beam_size}): {bleu_beam:.2f}")

    # If no beam-3, use the first beam size
    if 'bleu_beam' not in results:
        first_beam = config.BEAM_SIZES[1] if len(config.BEAM_SIZES) > 1 else 1
        results['bleu_beam'] = results.get(f'bleu_beam_{first_beam}', bleu_greedy)

    return results


def run_comprehensive_experiments():
    """Run comprehensive experiments comparing all configurations"""
    print("\n" + "=" * 80)
    print("RNN-based NMT - Comprehensive Experiments")
    print("Comparing:")
    print("  1. Attention mechanisms: dot, multiplicative, additive")
    print("  2. Training strategies: 100%, 50%, 0% teacher forcing")
    print("  3. Decoding strategies: greedy, beam search (1, 3, 5, 10)")
    print("=" * 80)

    all_results = []

    # First, train models with different attention mechanisms (using 50% teacher forcing)
    print("\n" + "=" * 80)
    print("PHASE 1: Comparing Attention Mechanisms")
    print("=" * 80)

    for attention_type in config.ATTENTION_TYPES:
        model_path = config.MODEL_PATHS[f'{attention_type}_50']
        results = train_and_evaluate(attention_type, teacher_forcing_ratio=0.5, model_path=model_path)
        all_results.append(results)

    # Then, compare teacher forcing strategies with best attention mechanism (additive)
    print("\n" + "=" * 80)
    print("PHASE 2: Comparing Training Strategies (Teacher Forcing vs Free Running)")
    print("=" * 80)

    for tf_ratio in [1.0, 0.0]:  # 100% and 0%
        model_path = f"{config.SAVE_DIR}/model_additive_{int(tf_ratio*100)}.pt"
        results = train_and_evaluate('additive', teacher_forcing_ratio=tf_ratio, model_path=model_path)
        all_results.append(results)

    # Save results
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    with open(config.EXPERIMENT_RESULTS_PATH, 'w') as f:
        f.write("RNN-based Neural Machine Translation - Experiment Results\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. ATTENTION MECHANISM COMPARISON (Teacher Forcing = 50%)\n")
        f.write("-" * 80 + "\n")
        for i, attention_type in enumerate(config.ATTENTION_TYPES):
            r = all_results[i]
            f.write(f"\n{attention_type.upper()} Attention:\n")
            f.write(f"  Valid Loss: {r['best_valid_loss']:.4f}\n")
            f.write(f"  Parameters: {r['n_params']:,}\n")
            f.write(f"  BLEU (Greedy): {r['bleu_greedy']:.2f}\n")
            f.write(f"  BLEU (Beam-3): {r.get('bleu_beam_3', 'N/A'):.2f}\n")
            f.write(f"  BLEU (Beam-5): {r.get('bleu_beam_5', 'N/A'):.2f}\n")
            f.write(f"  BLEU (Beam-10): {r.get('bleu_beam_10', 'N/A'):.2f}\n")

        f.write("\n\n2. TRAINING STRATEGY COMPARISON (Additive Attention)\n")
        f.write("-" * 80 + "\n")
        tf_ratios = [0.5, 1.0, 0.0]
        names = ['50% (Mixed)', '100% (Teacher Forcing)', '0% (Free Running)']
        for i, (tf_ratio, name) in enumerate(zip(tf_ratios, names)):
            if i < len(all_results):
                r = all_results[-(3-i)]
                f.write(f"\n{name}:\n")
                f.write(f"  Valid Loss: {r['best_valid_loss']:.4f}\n")
                f.write(f"  BLEU (Greedy): {r['bleu_greedy']:.2f}\n")
                f.write(f"  BLEU (Beam-3): {r.get('bleu_beam_3', 'N/A'):.2f}\n")
                f.write(f"  BLEU (Beam-5): {r.get('bleu_beam_5', 'N/A'):.2f}\n")
                f.write(f"  BLEU (Beam-10): {r.get('bleu_beam_10', 'N/A'):.2f}\n")

        f.write("\n\n3. DECODING STRATEGY COMPARISON (Best Model: Additive + 50% TF)\n")
        f.write("-" * 80 + "\n")
        best_result = all_results[2]  # Additive with 50% TF
        f.write(f"\nGreedy (Beam-1): BLEU = {best_result['bleu_greedy']:.2f}\n")
        f.write(f"Beam-3: BLEU = {best_result.get('bleu_beam_3', 'N/A'):.2f}\n")
        f.write(f"Beam-5: BLEU = {best_result.get('bleu_beam_5', 'N/A'):.2f}\n")
        f.write(f"Beam-10: BLEU = {best_result.get('bleu_beam_10', 'N/A'):.2f}\n")

        f.write("\n\n4. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")

        # Find best attention mechanism
        best_attention = max(all_results[:3], key=lambda x: x['bleu_greedy'])
        f.write(f"\nBest Attention Mechanism (Greedy): {best_attention['attention_type'].upper()}\n")

        # Find best training strategy
        best_training = max(all_results, key=lambda x: x['bleu_greedy'])
        f.write(f"Best Training Strategy: {best_training['teacher_forcing_ratio']*100:.0f}% Teacher Forcing\n")

        # Find best decoding strategy
        best_decoding = max(best_result.items(), key=lambda x: x[0].startswith('bleu') and isinstance(x[1], float))
        f.write(f"Best Decoding Strategy: {best_decoding[0].replace('bleu_', 'Beam-').replace('bleu_greedy', 'Greedy')} (BLEU: {best_decoding[1]:.2f})\n")

    print(f"\nResults saved to: {config.EXPERIMENT_RESULTS_PATH}")

    # Print summary to console
    print("\n📊 QUICK SUMMARY:")
    print(f"Best Attention: {best_attention['attention_type'].upper()} (BLEU: {best_attention['bleu_greedy']:.2f})")
    print(f"Best Training: {best_training['teacher_forcing_ratio']*100:.0f}% Teacher Forcing (BLEU: {best_training['bleu_greedy']:.2f})")
    print(f"Best Decoding: {best_decoding[0].replace('bleu_', '').replace('beam_', 'Beam-').replace('greedy', 'Greedy')} (BLEU: {best_decoding[1]:.2f})")

    print("\n" + "=" * 80)
    print("✓ All experiments completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_comprehensive_experiments()
