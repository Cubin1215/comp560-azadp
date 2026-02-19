import os
import pickle
import random
import numpy as np

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# How many examples to generate?
num_samples = 20000 
# Limit the number of digits (e.g., 3 means up to 999)
max_digits = 2

# -----------------------------------------------------------------------------
# 1. GENERATE THE DATASET
# -----------------------------------------------------------------------------
print(f"Generating {num_samples} addition problems...")

data = ""
# We will generate problems like "123+456=579\n"
for _ in range(num_samples):
    # Pick random number of digits for a and b (1 to max_digits)
    # We weight it so we get more hard problems (3-digit) than easy ones (1-digit)
    ndigits_a = random.randint(1, max_digits)
    ndigits_b = random.randint(1, max_digits)
    
    a = random.randint(0, 10**ndigits_a - 1)
    b = random.randint(0, 10**ndigits_b - 1)
    
    # Calculate correct answer
    c = a + b
    
    # Format the string: "a+b=c" plus a newline
    problem = f"{a}+{b}={c}\n"
    data += problem

# -----------------------------------------------------------------------------
# 2. BUILD VOCABULARY
# -----------------------------------------------------------------------------
# Get all unique characters in the data (should be 0-9, +, =, \n)
chars = sorted(list(set(data)))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")

# Create mappings (Character -> Integer and Integer -> Character)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# Encoder function: takes a string, outputs a list of integers
def encode(s):
    return [stoi[c] for c in s]

# Decoder function: takes a list of integers, outputs a string
def decode(l):
    return ''.join([itos[i] for i in l])

# -----------------------------------------------------------------------------
# 3. SAVE BINARY FILES
# -----------------------------------------------------------------------------
# Split into Train (90%) and Validation (10%)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Encode both sets
train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Save to binary files (uint16 is enough for small vocabs)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save the meta information (vocabulary) so we can decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Saved meta.pkl, train.bin, and val.bin!")