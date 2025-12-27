# RNN NMT Inference Guide

This guide explains how to use the inference script (`inference.py`) to perform translations with your trained RNN-based Neural Machine Translation models.

## Prerequisites

1. **Trained Model**: You must have a trained model checkpoint
2. **Vocabularies**: Make sure vocabulary files exist in the `vocab/` directory
3. **Dependencies**: Install all required packages (torch, jieba, sentencepiece, etc.)

## Quick Start

### 1. Basic Translation

```bash
# Chinese to English
python inference.py \
    --checkpoint checkpoints/10k/model_dot_best.pt \
    --direction zh2en \
    --dataset 10k \
    --attention dot \
    -- "我爱学习自然语言处理"

# English to Chinese
python inference.py \
    --checkpoint checkpoints/10k/model_dot_best.pt \
    --direction en2zh \
    --dataset 10k \
    --attention dot \
    -- "I love studying natural language processing"
```

### 2. Interactive Mode

```bash
python inference.py \
    --checkpoint checkpoints/10k/model_dot_best.pt \
    --direction zh2en \
    --dataset 10k \
    --attention dot \
    --beam_size 5 \
    --interactive
```

In interactive mode, you can type sentences to translate one by one. Type `quit` or `exit` to stop.

### 3. Batch Translation

```bash
# Create input file
echo -e "今天天气很好\n我喜欢吃苹果\n机器学习很有趣" > chinese_sentences.txt

# Translate all sentences
python inference.py \
    --checkpoint checkpoints/10k/model_dot_best.pt \
    --direction zh2en \
    --dataset 10k \
    --attention dot \
    --beam_size 5 \
    --input_file chinese_sentences.txt \
    --output_file english_translations.txt
```

## Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--checkpoint` | str | Yes | - | Path to model checkpoint (.pt file) |
| `--direction` | str | Yes | - | Translation direction: `zh2en` or `en2zh` |
| `--dataset` | str | No | `10k` | Dataset size: `10k` or `100k` |
| `--attention` | str | No | `dot` | Attention type: `dot`, `multiplicative`, `additive` |
| `--beam_size` | int | No | `5` | Beam size (1 = greedy decoding) |
| `--max_length` | int | No | `100` | Maximum translation length |
| `--length_penalty` | float | No | `0.6` | Length penalty for beam search |
| `--input_file` | str | No | - | Input file for batch translation |
| `--output_file` | str | No | auto-generated | Output file for batch translation |
| `--interactive` | flag | No | - | Run in interactive mode |
| `--vocab_dir` | str | No | `vocab` | Directory containing vocabularies |

## Model Checkpoints

The inference script expects model checkpoints in the following format:

```
checkpoints/
├── 10k/
│   ├── model_dot_best.pt
│   ├── model_multiplicative_best.pt
│   └── model_additive_best.pt
└── 100k/
    ├── model_dot_best.pt
    ├── model_multiplicative_best.pt
    └── model_additive_best.pt
```

## Translation Directions

### Chinese to English (zh2en)
- Input: Chinese characters (e.g., "我爱学习")
- Output: English text (e.g., "I love studying")
- Uses jieba for Chinese tokenization
- Uses SentencePiece for English detokenization

### English to Chinese (en2zh)
- Input: English text (e.g., "Hello world")
- Output: Chinese characters (e.g., "你好世界")
- Uses SentencePiece for English tokenization
- Combines tokens directly for Chinese output

## Decoding Options

### Greedy Decoding
```bash
--beam_size 1
```
- Fastest option
- Selects the most likely token at each step
- Good for quick tests

### Beam Search
```bash
--beam_size 5  # or 3, 10, etc.
```
- Maintains multiple hypotheses
- Generally produces better translations
- Larger beam sizes = better quality but slower

### Length Penalty
```bash
--length_penalty 0.6
```
- Controls preference for shorter/longer translations
- Values < 1.0 favor shorter sequences
- Values > 1.0 favor longer sequences

## Examples

### Example 1: Comparing Attention Types
```bash
for attention in dot multiplicative additive; do
    python inference.py \
        --checkpoint checkpoints/10k/model_${attention}_best.pt \
        --direction zh2en \
        --attention $attention \
        --beam_size 5 \
        -- "这是一个测试句子"
done
```

### Example 2: Comparing Beam Sizes
```bash
for beam in 1 3 5 10; do
    echo -e "\nBeam size: $beam"
    python inference.py \
        --checkpoint checkpoints/10k/model_dot_best.pt \
        --direction zh2en \
        --attention dot \
        --beam_size $beam \
        -- "深度学习改变了自然语言处理"
done
```

### Example 3: Large Scale Translation
```bash
# For translating a large file
python inference.py \
    --checkpoint checkpoints/100k/model_dot_best.pt \
    --direction zh2en \
    --dataset 100k \
    --attention dot \
    --beam_size 5 \
    --max_length 150 \
    --input_file large_chinese_corpus.txt \
    --output_file english_translations.txt
```

## Tips

1. **Memory Usage**: For long documents or large files, consider using smaller batch sizes
2. **Speed**: Greedy decoding (beam_size=1) is fastest, but beam search produces better results
3. **Quality**: The best configuration depends on your trained model and language pair
4. **Interactive Mode**: Perfect for testing and debugging individual sentences
5. **Batch Processing**: Use file input/output for processing multiple sentences efficiently

## Troubleshooting

### Common Errors

1. **FileNotFoundError**: Check that your checkpoint path is correct
2. **Vocabulary not found**: Ensure vocab files exist in `vocab/{dataset}/`
3. **CUDA out of memory**: Try using CPU or reducing beam size
4. **Tokenization errors**: Make sure the correct dataset (10k/100k) is specified

### Debug Mode

Add print statements to see intermediate results:
```python
# In inference.py, after preprocessing
print(f"Source tokens: {tokens}")
print(f"Source indices: {src_indices}")
```

## Performance

Typical translation speeds (on a V100 GPU):
- Greedy decoding: ~100 sentences/second
- Beam size 5: ~20 sentences/second
- Beam size 10: ~10 sentences/second

CPU speeds are approximately 10x slower.

## Next Steps

1. **Evaluate**: Calculate BLEU scores on test sets
2. **Fine-tune**: Adjust beam size and length penalty
3. **Optimize**: Use quantization or pruning for faster inference
4. **Deploy**: Wrap in a REST API for production use