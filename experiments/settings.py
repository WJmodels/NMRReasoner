"""Training defaults inherited from the original experiment.

Edit these values for advanced experiments. The CLI overrides data, model,
epochs and per-device batch size via the existing REASONER_* environment keys.
"""

MAX_SEQ_LENGTH = 5000
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 0.15
WARMUP_STEPS = 5
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
SAVE_STEPS = 8000
SAVE_TOTAL_LIMIT = 300
FULL_FINETUNING = True
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
LORA_R = 512
LORA_ALPHA = 2048
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
TRAIN_PROMPT_STYLE = "{}<think>{}</think><answer>{}</answer>"
