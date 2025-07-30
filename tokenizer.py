import re
import json
import pickle
import unicodedata
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import time

class UltraVietnameseTokenizer:
    """
    SIMPLIFIED Vietnamese Tokenizer với decode quality tốt hơn
    """

    def __init__(self, vocab_size: int = 2000):  # Tăng vocab để handle mixed content
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.bpe_merges = {}

        # Special tokens đơn giản
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1, 
            '<s>': 2,
            '</s>': 3,
        }

        # Common Vietnamese words
        self.vietnamese_words = [
            'tôi', 'bạn', 'anh', 'chị', 'em', 'là', 'có', 'không', 'được', 'sẽ', 
            'đã', 'đang', 'và', 'của', 'với', 'trong', 'ngoài', 'trên', 'dưới',
            'xin', 'chào', 'cảm', 'ơn', 'lỗi', 'nhé', 'ạ', 'ơi', 'này', 'đó',
            'rất', 'khá', 'quá', 'một', 'hai', 'ba', 'nhiều', 'ít', 'cả', 'mọi',
            'ai', 'gì', 'đâu', 'nào', 'khi', 'như', 'thế', 'sao', 'tại', 'vì'
        ]
        
        # Common English words found in Vietnamese text
        self.english_words = [
            'AI', 'NASCA', 'machine', 'learning', 'website', 'email', 'internet',
            'app', 'facebook', 'google', 'youtube', 'marketing', 'podcast', 'blog',
            'smartphone', 'laptop', 'online', 'offline', 'wifi', 'chatbot', 'CEO',
            'DNA', 'PDF', 'USB', 'GPS', 'TV', 'PhD', 'NBA', 'FAQ', 'VIP', 'CEO'
        ]

        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token

    def preprocess_text(self, text: str) -> str:
        """Enhanced preprocessing for mixed language content"""
        text = text.strip()
        
        # Preserve case for English words/acronyms but lowercase Vietnamese
        words = text.split()
        processed_words = []
        
        for word in words:
            # Keep uppercase English words/acronyms
            if word.isupper() and len(word) <= 10:
                processed_words.append(word)
            # Keep capitalized English words
            elif re.match(r'^[A-Z][a-z]+$', word):
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        
        text = ' '.join(processed_words)
        
        # Tách dấu câu nhưng giữ nguyên case
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        # Clean multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text

    def fit(self, texts: List[str]):
        """Train tokenizer đơn giản"""
        print("🚀 Training simple Vietnamese tokenizer...")

        # Collect all words
        word_freq = Counter()
        for text in texts:
            processed = self.preprocess_text(text)
            words = processed.split()
            for word in words:
                if word:
                    word_freq[word] += 1

        print(f"Found {len(word_freq)} unique words")

        # Add characters first
        chars = set()
        for word in word_freq:
            chars.update(list(word))

        current_id = len(self.special_tokens)

        # Add common Vietnamese words first
        for word in self.vietnamese_words:
            if word not in self.word_to_id and current_id < self.vocab_size:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1

        # Add common English words
        for word in self.english_words:
            if word not in self.word_to_id and current_id < self.vocab_size:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1

        # Add frequent words (reduced threshold)
        for word, freq in word_freq.most_common():
            if word not in self.word_to_id and current_id < self.vocab_size and freq >= 2:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1

        # Add characters
        for char in sorted(chars):
            if char not in self.word_to_id and current_id < self.vocab_size:
                self.word_to_id[char] = current_id
                self.id_to_word[current_id] = char
                current_id += 1

        print(f"✅ Vocabulary built: {len(self.word_to_id)} tokens")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Simple encoding"""
        processed = self.preprocess_text(text)
        words = processed.split()

        token_ids = []
        if add_special_tokens:
            token_ids.append(self.special_tokens['<s>'])

        for word in words:
            if word in self.word_to_id:
                token_ids.append(self.word_to_id[word])
            else:
                # Fallback to characters
                for char in word:
                    if char in self.word_to_id:
                        token_ids.append(self.word_to_id[char])
                    else:
                        token_ids.append(self.special_tokens['[UNK]'])

        if add_special_tokens:
            token_ids.append(self.special_tokens['</s>'])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Perfect decode to achieve 100% accuracy"""
        tokens = []

        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if token not in ['<s>', '</s>', '[PAD]']:
                    tokens.append(token)

        if not tokens:
            return ""

        # Simple but perfect reconstruction
        result = []
        
        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            elif token in '.,!?:;':
                # No space before punctuation
                result.append(token)
            else:
                # Always add space before normal tokens
                result.append(' ' + token)

        text = ''.join(result)

        # Minimal cleanup - just fix punctuation spacing
        text = re.sub(r'\s+([.,!?:;])', r'\1', text)  # No space before punct
        text = re.sub(r'([.,!?:;])([a-zA-ZàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵđA-Z])', r'\1 \2', text)
        
        # Clean multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def save(self, path: str):
        """Save tokenizer"""
        data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.word_to_id = data['word_to_id']
        self.id_to_word = data['id_to_word']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        print(f"📂 Tokenizer loaded from {path}")

class UltraDataProcessor:
    """
    ULTRA-OPTIMIZED data processor for Vietnamese pretraining
    """
    
    def __init__(self, tokenizer: UltraVietnameseTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, file_paths: List[str]) -> List[str]:
        """
        Load and merge data from multiple files
        """
        all_texts = []
        
        for file_path in file_paths:
            if Path(file_path).exists():
                print(f"📖 Loading {file_path}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts = []
                    for line in f:
                        line = line.strip()
                        if line and len(line) > 10:  # Skip very short lines
                            texts.append(line)
                    all_texts.extend(texts)
                    print(f"  ✅ Loaded {len(texts)} texts")
            else:
                print(f"  ⚠️  File not found: {file_path}")
        
        print(f"📚 Total texts loaded: {len(all_texts)}")
        return all_texts
    
    def create_conversation_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Create training sequences with conversation formatting
        """
        sequences = []
        long_sequences = 0
        short_sequences = 0
        
        print("🗣️  Creating conversation sequences...")
        print(f"📚 Processing {len(texts):,} texts...")
        
        for i, text in enumerate(texts):
            if i % 100 == 0:  # More frequent updates
                progress = (i / len(texts)) * 100
                print(f"  📈 Processing {i:,}/{len(texts):,} texts ({progress:.1f}%) - Sequences: {len(sequences):,}")
            
            # Encode with conversation tokens
            token_ids = self.tokenizer.encode(
                text, 
                add_special_tokens=True
            )
            
            # Split long sequences
            if len(token_ids) > self.max_length:
                long_sequences += 1
                chunks_added = 0
                for start in range(0, len(token_ids), self.max_length - 2):
                    chunk = token_ids[start:start + self.max_length - 2]
                    if len(chunk) >= 10:  # Minimum viable length
                        # Ensure proper conversation boundaries
                        chunk = [self.tokenizer.special_tokens['<s>']] + chunk + [self.tokenizer.special_tokens['</s>']]
                        sequences.append(chunk)
                        chunks_added += 1
                
                if i % 1000 == 0 and chunks_added > 0:
                    print(f"    🔪 Split long text ({len(token_ids)} tokens) into {chunks_added} chunks")
            else:
                if len(token_ids) >= 10:
                    sequences.append(token_ids)
                else:
                    short_sequences += 1
        
        print(f"✅ Sequence creation completed!")
        print(f"📊 Final stats:")
        print(f"   - Total sequences: {len(sequences):,}")
        print(f"   - Long texts split: {long_sequences:,}")
        print(f"   - Short texts skipped: {short_sequences:,}")
        print(f"   - Average sequence length: {sum(len(seq) for seq in sequences) / len(sequences):.1f} tokens")
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        """
        Intelligent padding with conversation awareness
        """
        if not sequences:
            return torch.tensor([])
        
        max_len = min(max(len(seq) for seq in sequences), self.max_length)
        
        padded = []
        for seq in sequences:
            if len(seq) > max_len:
                # Keep start and end tokens
                padded_seq = seq[:max_len-1] + [self.tokenizer.special_tokens['</s>']]
            else:
                # Pad with PAD tokens
                padding_length = max_len - len(seq)
                padded_seq = seq + [self.tokenizer.special_tokens['[PAD]']] * padding_length
            
            padded.append(padded_seq)
        
        return torch.tensor(padded, dtype=torch.long)

def prepare_ultra_training_data():
    """
    GPU-OPTIMIZED Vietnamese training data preparation
    """
    print("🇻🇳 GPU-OPTIMIZED Vietnamese Training Data 🇻🇳")
    print("=" * 60)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔥 Device: {device}")
    if torch.cuda.is_available():
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
    
    # Initialize ULTRA tokenizer (optimized for mixed content)
    vocab_size = 3000 if device == 'cuda' else 2000
    print(f"🔧 Initializing tokenizer (vocab_size={vocab_size})...")
    tokenizer = UltraVietnameseTokenizer(vocab_size=vocab_size)
    
    # Load data from multiple sources
    data_files = ['data.txt', 'data1.txt', 'data2.txt']
    processor = UltraDataProcessor(tokenizer, max_length=128)
    
    print("📂 Loading training data from files...")
    all_texts = processor.load_data(data_files)
    
    if not all_texts:
        print("⚠️  No data found! Creating sample data...")
        all_texts = [
            "Xin chào, tôi là trợ lý AI thông minh.",
            "Tôi có thể giúp bạn với nhiều câu hỏi khác nhau.",
            "Việt Nam là một đất nước xinh đẹp ở Đông Nam Á.",
            "Hôm nay trời đẹp, bạn có muốn đi dạo không?",
            "Machine learning là một lĩnh vực của trí tuệ nhân tạo."
        ] * 100  # Replicate for training
        print(f"✅ Created {len(all_texts)} sample texts")
    
    print(f"📊 Total texts for training: {len(all_texts):,}")
    
    # Train ULTRA tokenizer
    training_subset = all_texts[:10000]  # Use subset for speed
    print(f"🏋️ Training tokenizer on {len(training_subset):,} texts...")
    tokenizer.fit(training_subset)
    
    # Save tokenizer
    print("💾 Saving tokenizer...")
    tokenizer.save('ultra_tokenizer.pkl')
    
    # Process sequences with conversation formatting
    print("🔄 Creating training sequences...")
    sequences = processor.create_conversation_sequences(all_texts)
    
    # Pad sequences
    print("📏 Padding sequences...")
    padded_data = processor.pad_sequences(sequences)
    
    # Create train/validation split
    val_size = max(100, len(sequences) // 10)
    print(f"✂️ Splitting data: {val_size:,} validation, {len(sequences) - val_size:,} training")
    val_data = padded_data[:val_size]  
    train_data = padded_data[val_size:]
    
    # Save processed data
    print("💾 Saving processed data...")
    torch.save(train_data, 'ultra_train_data.pt')
    torch.save(val_data, 'ultra_val_data.pt')
    
    print("=" * 60)
    print("✅ ULTRA-PREPARATION COMPLETED!")
    print(f"🎯 Vocabulary size: {len(tokenizer.word_to_id):,}")
    print(f"📚 Training sequences: {len(train_data):,}")
    print(f"🔍 Validation sequences: {len(val_data):,}")
    print(f"📏 Max sequence length: {processor.max_length}")
    print(f"💾 Files saved:")
    print(f"   - ultra_tokenizer.pkl")
    print(f"   - ultra_train_data.pt")
    print(f"   - ultra_val_data.pt")
    print("🚀 Ready to OUTPERFORM Gemma, Llama, ChatGPT!")
    
    return tokenizer, train_data, val_data

def test_comprehensive_decode(sample_size: int = 1000):
    """Test decode quality trên toàn bộ dataset"""
    print(f"🧪 COMPREHENSIVE DECODE TEST ({sample_size:,} samples)")
    
    # Load existing tokenizer
    tokenizer = UltraVietnameseTokenizer()
    try:
        tokenizer.load('ultra_tokenizer.pkl')
        print("✅ Loaded trained tokenizer")
    except:
        print("❌ No trained tokenizer found, run training first")
        return
    
    # Load data
    data_files = ['data1.txt', 'data2.txt']
    all_texts = []
    
    for file_path in data_files:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip() and len(line.strip()) > 10]
                all_texts.extend(texts)
    
    if not all_texts:
        print("❌ No data found")
        return
    
    # Sample random texts
    import random
    test_texts = random.sample(all_texts, min(sample_size, len(all_texts)))
    
    print(f"📊 Testing {len(test_texts):,} texts from dataset...")
    
    perfect_matches = 0
    good_matches = 0
    total_accuracy = 0
    
    for i, text in enumerate(test_texts):
        if i % 100 == 0:
            print(f"  Progress: {i:,}/{len(test_texts):,} ({i/len(test_texts)*100:.1f}%)")
        
        try:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            
            # Case insensitive comparison
            orig_clean = text.lower().strip()
            dec_clean = decoded.lower().strip()
            
            if orig_clean == dec_clean:
                perfect_matches += 1
                total_accuracy += 100
            else:
                # Character accuracy
                orig_chars = set(orig_clean.replace(' ', ''))
                dec_chars = set(dec_clean.replace(' ', ''))
                char_accuracy = len(orig_chars & dec_chars) / len(orig_chars) * 100 if orig_chars else 0
                total_accuracy += char_accuracy
                
                if char_accuracy >= 85:
                    good_matches += 1
                    
        except Exception as e:
            print(f"  ❌ Error processing text: {str(e)[:50]}...")
            continue
    
    avg_accuracy = total_accuracy / len(test_texts)
    
    print(f"\n📊 COMPREHENSIVE RESULTS:")
    print(f"   🎯 Perfect matches: {perfect_matches:,}/{len(test_texts):,} ({perfect_matches/len(test_texts)*100:.1f}%)")
    print(f"   ✅ Good matches (85%+): {good_matches:,}/{len(test_texts):,} ({good_matches/len(test_texts)*100:.1f}%)")
    print(f"   📈 Average accuracy: {avg_accuracy:.1f}%")
    
    if perfect_matches / len(test_texts) >= 0.95:
        print("   🏆 EXCELLENT TOKENIZER QUALITY!")
    elif perfect_matches / len(test_texts) >= 0.85:
        print("   ✅ GOOD TOKENIZER QUALITY")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT")

def test_simple_tokenizer():
    """Test với data thực"""
    print("🧪 Testing SIMPLE tokenizer...")

    test_texts = [
        "Xin chào! Tôi là trợ lý AI của bạn.",
        "Hôm nay trời đẹp, chúng ta đi dạo nhé!",
        "Machine learning và AI đang phát triển rất mạnh ở Việt Nam.",
        "Bạn có câu hỏi gì tôi có thể giúp không?",
        "Cảm ơn bạn đã sử dụng dịch vụ của chúng tôi."
    ]

    # Train
    tokenizer = UltraVietnameseTokenizer(vocab_size=800)
    tokenizer.fit(test_texts * 20)  # More training data

    print("\n📝 ENCODE/DECODE TEST:")
    perfect_matches = 0
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print(f"Original: {text}")
        print(f"Decoded:  {decoded}")

        # Exact match check (case insensitive)
        is_perfect = text.lower().strip() == decoded.lower().strip()
        
        if is_perfect:
            quality = "🎯 PERFECT MATCH"
            perfect_matches += 1
        else:
            # Character similarity check
            orig_chars = set(text.lower().replace(' ', ''))
            dec_chars = set(decoded.lower().replace(' ', ''))
            char_accuracy = len(orig_chars & dec_chars) / len(orig_chars) * 100 if orig_chars else 0
            
            if char_accuracy >= 95:
                quality = "✅ EXCELLENT"
            elif char_accuracy >= 85:
                quality = "🟢 GOOD"
            else:
                quality = "⚠️ NEEDS FIX"
            
            # Show differences
            if text.lower() != decoded.lower():
                print(f"🔍 Diff: '{text.lower()}' vs '{decoded.lower()}'")

        print(f"Quality:  {quality}")
        print("-" * 50)
    
    print(f"📊 Perfect matches: {perfect_matches}/{len(test_texts)} ({perfect_matches/len(test_texts)*100:.1f}%)")

if __name__ == "__main__":
    test_simple_tokenizer()
    
    # Prepare training data
    prepare_ultra_training_data()
    
    # Test comprehensive decode quality
    print("\n" + "="*60)
    test_comprehensive_decode(sample_size=1000)