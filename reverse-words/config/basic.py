# Train a small model on the "basic" dataset
out_dir = 'out'
eval_interval = 50 # check validation loss every 50 steps
eval_iters = 20
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'reverse-words'
wandb_run_name = 'mini-gpt'

dataset = 'basic'  # This points to data/basic/
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64 # context of up to 64 characters

# tiny GPT model parameters
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3
max_iters = 200 # Start small as requested
lr_decay_iters = 2000 # make equal to max_iters usually
min_lr = 1e-4 
beta2 = 0.99 

warmup_iters = 100