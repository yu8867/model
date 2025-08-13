import torch

class ModelConfig:
    epoch_num = 100
    token = 64
    batch_size = 96
    hidden_size = 256
    n_block = 4
    n_head = 8
    dropout_rate = 0.1
    save_folder = "GPT2_2/results"
    dataset_path = 'aozorabunko.txt'
    model_path = 'GPT2_2/model'
    tokenizer_path = 'GPT2_2/tokenizer'
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")