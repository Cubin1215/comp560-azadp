# Train a small model for addition
# -----------------------------------------------------------------------------

out_dir = 'out-addition'
eval_interval = 50 # how often to check if it's learning
eval_iters = 20
log_interval = 10 # print progress every 10 steps

# always save a checkpoint when validation loss improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'addition_cot'
wandb_run_name = 'mini-gpt'

dataset = 'addition'
batch_size = 64
block_size = 64 # context of up to 256 characters

# Baby GPT model specifications
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 200
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary for potentially small model, but good practice

# on macbook usually add device='cpu' if no gpu available
# device = 'cpu'  # uncomment this if you get CUDA errors on a laptop without NVIDIA GPU
compile = False # do not torch compile the model (difficult to debug)