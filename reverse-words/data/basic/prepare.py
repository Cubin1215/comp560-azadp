import os
import pickle
import random
import numpy as np

# --- 1. Generate the Data Stream (Reverse Words) ---
def generate_data(num_lines=10000):
    chars = "abcdefghijklmnopqrstuvwxyz"
    data = []
    for _ in range(num_lines):
        length = random.randint(3, 8)
        word = "".join(random.choices(chars, k=length))
        # The pattern: word + ":" + reversed_word + newline
        line = f"{word}: {word[::-1]}\n"
        data.append(line)
    return "".join(data)

print("Generating random data...")
raw_data = generate_data()
print(f"Sample data:\n{raw_data[:50]}...")

# --- 2. Build Vocabulary ---
chars = sorted(list(set(raw_data)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

# --- 3. Save Meta Data (Pickle) ---
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# --- 4. Create Train/Val Splits ---
n = len(raw_data)
train_data = raw_data[:int(n*0.9)]
val_data = raw_data[int(n*0.9):]

# Encode
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"Train tokens: {len(train_ids)}, Val tokens: {len(val_ids)}")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print("Done! Files saved to data/basic/")