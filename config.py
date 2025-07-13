import torch

class DefaultConfig(object):
    train_dataset_path = r'data\\train355-r5.5-a2.3.pkl'
    test_dataset_path = r'data\\Test60.pkl'
    save_path = r'Model_saved'

    epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-4
    dropout_rate = 0.2
    split_rate = 0.8
    batch_size = 24
    temperature = 0.07

    # Pre_train
    low_hid_dim=128
    high_hid_dim=256
    scenario_dim=256
    high_out_dim=128

    # mlp
    mlp_dim = 128
    mlp_hid_dim = 128
    mlp_out_dim = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seeds = [649737]