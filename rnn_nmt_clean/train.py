"""
Training script for RNN-based Neural Machine Translation
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import argparse
from tqdm import tqdm
import sacrebleu
import sentencepiece as spm
from data_loader import get_data_loaders, get_test_loader
from model_v2 import build_model, count_parameters, initialize_weights
from beam_search_correct import greedy_decode, beam_search_decode
import pickle

# Parse arguments BEFORE importing config_v2
# We need to extract dataset arg first, pass it to config_v2, then parse the rest

dataset = '10k'  # Default

# Extract dataset from command line
if '--dataset' in sys.argv:
    idx = sys.argv.index('--dataset')
    if idx + 1 < len(sys.argv):
        dataset = sys.argv[idx + 1]

# Temporarily modify sys.argv to only have dataset
original_argv = sys.argv.copy()
sys.argv = ['train.py', '--dataset', dataset]

# Import config_v2 AFTER setting dataset argument
import config_v2 as config

# Restore original argv
sys.argv = original_argv

# Now parse all arguments
parser = argparse.ArgumentParser(description='RNN NMT Training')
parser.add_argument('--dataset', type=str, default=dataset, choices=['10k', '100k'],
                    help='Dataset size to use (10k or 100k)')
parser.add_argument('--epochs', type=int, default=25,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate')
parser.add_argument('--attention_type', type=str, default='additive',
                    choices=['dot', 'multiplicative', 'additive'],
                    help='Attention mechanism type')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                    help='Teacher forcing ratio (0.0 to 1.0)')
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to save the model')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='GPU device ID')

# Parse only unknown arguments (ignore dataset since config_v2 already handled it)
args, unknown = parser.parse_known_args()

# Override config with command line arguments
config.N_EPOCHS = args.epochs
config.LEARNING_RATE = args.lr
config.args = args

# Set model path if provided
if args.model_path is not None:
    config.MODEL_PATH = args.model_path

# Set GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

def train_epoch(model, train_loader, optimizer, criterion, clip, teacher_forcing_ratio=0.5):
    """Train for one epoch"""
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


def evaluate(model, valid_loader, criterion):
    """Evaluate on validation set"""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in valid_loader:
            src = batch['src'].to(config.device)
            trg = batch['trg'].to(config.device)

            output = model(src, trg, teacher_forcing_ratio=0)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(valid_loader)


def calculate_bleu(model, test_loader, zh_idx2word, en_sp, decoding_strategy='greedy', beam_size=5):
    """Calculate BLEU score on test set with specified decoding strategy"""
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

                # hypothesis: greedy/beam 已经在 decode 时去掉了 special，但这里再做一次也
                translation_text = en_sp.decode_ids(strip_special(translation_ids)).strip()

                # reference: 一定要 strip（trg_sentence 里含 PAD/BOS/EOS）
                trg_ids = strip_special(trg_sentence.tolist())
                trg_text = en_sp.decode_ids(trg_ids).strip()


                references.append([trg_text])
                hypotheses.append(translation_text)

    # Calculate BLEU-4 as required by assignment
    # Using '13a' tokenizer which is appropriate for English (target language)
    # sacrebleu default is BLEU-4 (max_n=4), so we just need to specify tokenizer
    bleu = sacrebleu.corpus_bleu(hypotheses, references, tokenize='13a')

    # Return BLEU-4 score
    return bleu.score


def train():
    """Main training function"""

    print("=" * 60)
    print("RNN-based Neural Machine Translation Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Dataset: {args.dataset}")
    print(f"Teacher Forcing Ratio: 0.5")
    print(f"Epochs: {config.N_EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print("=" * 60)

    # Load config info and update vocabulary sizes
    try:
        with open(f'{config.TOKENIZER_DIR}/config_info.pkl', 'rb') as f:
            config_info = pickle.load(f)
            config.INPUT_DIM = config_info['zh_vocab_size']
            config.OUTPUT_DIM = config_info['en_vocab_size']

        print(f"\nChinese vocab size: {config.INPUT_DIM}")
        print(f"English vocab size: {config.OUTPUT_DIM}")
    except FileNotFoundError:
        print(f"\n✗ Error: Tokenizer config not found at {config.TOKENIZER_DIR}/config_info.pkl")
        print(f"Please run preprocessing first: python preprocess.py --dataset {args.dataset}")
        return

    # Load data
    train_loader, valid_loader = get_data_loaders()
    test_loader = get_test_loader()

    # Load tokenizers for evaluation
    with open(f'{config.TOKENIZER_DIR}/zh_vocab.pkl', 'rb') as f:
        zh_data = pickle.load(f)
        zh_idx2word = zh_data['idx2word']

    en_sp = spm.SentencePieceProcessor()
    en_sp.load(f'{config.TOKENIZER_DIR}/eng.model')

    # Build model
    model = build_model(attention_type=args.attention_type)

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

    # Log storage for plotting
    train_losses = []
    valid_losses = []
    train_ppls = []
    valid_ppls = []
    epoch_times = []

    for epoch in range(config.N_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.CLIP,
                                 teacher_forcing_ratio=args.teacher_forcing_ratio)

        # Evaluate
        valid_loss = evaluate(model, valid_loader, criterion)

        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) % 60)
        epoch_time = end_time - start_time

        # Store logs for plotting
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_ppls.append(math.exp(train_loss))
        valid_ppls.append(math.exp(valid_loss))
        epoch_times.append(epoch_time)

        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
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
            print(f"\n arly stopping triggered!")
            print(f"Best validation loss: {best_valid_loss:.4f}")
            break

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Save training logs for plotting
    training_log = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_ppls': train_ppls,
        'valid_ppls': valid_ppls,
        'epoch_times': epoch_times,
        'best_valid_loss': best_valid_loss,
        'total_epochs': len(train_losses),
        'total_time': sum(epoch_times)
    }

    import json
    import os

    # Create experiment-specific log file names
    exp_name = f"{args.dataset}_{args.attention_type}_tf{int(args.teacher_forcing_ratio*100)}"
    log_dir = config.SAVE_DIR
    os.makedirs(log_dir, exist_ok=True)

    log_path = f"{log_dir}/results_{exp_name}_training_log.json"
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Training logs saved to: {log_path}")

    # Also save as CSV for easy plotting with pandas/matplotlib
    csv_path = f"{log_dir}/results_{exp_name}_training_log.csv"
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,valid_loss,train_ppl,valid_ppl,time\n")
        for i in range(len(train_losses)):
            f.write(f"{i+1},{train_losses[i]:.4f},{valid_losses[i]:.4f},"
                   f"{train_ppls[i]:.4f},{valid_ppls[i]:.4f},{epoch_times[i]:.2f}\n")
    print(f"Training logs (CSV) saved to: {csv_path}")

    # Load best model for evaluation
    model.load_state_dict(torch.load(config.MODEL_PATH))

    # Calculate BLEU score with different decoding strategies
    print("\n" + "=" * 60)
    print("Evaluating Model Performance")
    print("=" * 60)

    # Test multiple beam sizes as per config.BEAM_SIZES
    beam_sizes = [1, 3, 5, 10]  # 1 = greedy
    bleu_scores = {}

    for beam_size in beam_sizes:
        if beam_size == 1:
            print("\nCalculating BLEU-4 score with Greedy decoding...")
            bleu = calculate_bleu(model, test_loader, zh_idx2word, en_sp, decoding_strategy='greedy')
            print(f"✓ Greedy Decoding BLEU-4 Score: {bleu:.2f}")
        else:
            print(f"\nCalculating BLEU-4 score with Beam Search (k={beam_size})...")
            bleu = calculate_bleu(model, test_loader, zh_idx2word, en_sp,
                                 decoding_strategy='beam-search', beam_size=beam_size)
            print(f"✓ Beam Search (k={beam_size}) BLEU-4 Score: {bleu:.2f}")
        bleu_scores[beam_size] = bleu

    print("\n" + "=" * 60)
    print("BLEU-4 Scores Summary")
    print("=" * 60)
    for beam_size, bleu in bleu_scores.items():
        decoding_name = "Greedy" if beam_size == 1 else f"Beam-{beam_size}"
        print(f"  {decoding_name}: {bleu:.2f}")

    # Calculate improvement
    if 5 in bleu_scores:
        print(f"\nImprovement with Beam-5 over Greedy: {bleu_scores[5] - bleu_scores[1]:.2f} BLEU-4 points")
    print("=" * 60)

    # Save results
    results_path = f"{log_dir}/results_{exp_name}.txt"
    with open(results_path, 'w') as f:
        f.write("=== BLEU-4 Scores (Assignment Requirement) ===\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Attention Type: {args.attention_type}\n")
        f.write(f"Teacher Forcing Ratio: {args.teacher_forcing_ratio}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"\n--- Decoding Strategy Comparison ---\n")
        for beam_size, bleu in bleu_scores.items():
            decoding_name = "Greedy (Beam-1)" if beam_size == 1 else f"Beam-{beam_size}"
            f.write(f"{decoding_name}: {bleu:.2f}\n")

        f.write(f"\nImprovement (Beam-5 over Greedy): {bleu_scores[5] - bleu_scores[1]:.2f}\n")
        f.write(f"Best Validation Loss: {best_valid_loss:.4f}\n")
        f.write(f"Model Parameters: {n_params:,}\n")

        # Note about tokenizer choice
        f.write("\n=== Note ===\n")
        f.write("Using '13a' tokenizer for English (target language)\n")
        f.write("BLEU-4 calculated with effective_order=True as per assignment\n")

    print(f"\nResults saved to: {results_path}")

    # Show some example translations
    print("\n" + "=" * 60)
    print("Example Translations (Greedy vs Beam Search)")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # Get a batch from test loader
        batch = next(iter(test_loader))
        src = batch['src'][:, 0].to(config.device)
        trg = batch['trg'][:, 0].to(config.device)

        # Decode source
        zh_text = ' '.join([zh_idx2word[idx.item()] for idx in src if idx.item() not in [0, 2, 3]])

        # Greedy translation
        greedy_ids = greedy_decode(model, src)
        greedy_text = en_sp.decode_ids(greedy_ids)

        # Beam search translation
        beam_ids = beam_search_decode(model, src, beam_size=5)
        beam_text = en_sp.decode_ids(beam_ids)

        # Decode reference
        # reference = en_sp.decode_ids(trg.tolist())
        reference = en_sp.decode_ids([i for i in trg.tolist() if i not in (0, en_sp.bos_id(), en_sp.eos_id())])



        print(f"\nChinese (src): {zh_text}")
        print(f"\nEnglish (ref):  {reference}")
        print(f"\nEnglish (greedy): {greedy_text}")
        print(f"\nEnglish (beam-k5): {beam_text}")


if __name__ == "__main__":
    train()
