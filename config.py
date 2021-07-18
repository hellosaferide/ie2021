
train_config = {
    "hyper_parameters": {
        "batch_size": 12,
        "epochs": 20,
        "learning_rate": 1e-5,
        "max_len": 512,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-5,
        "rewarm_epoch_num": 5,
        "T_mult": 1,
        "output_dir": "save_model",
        "train_path": 'data/duie_train.json',
        "valid_path": 'data/duie_dev.json',
        "weight_decay": 0.1,
        "model_path": "pretrained_model/RoBERTa_zh",
        "task_learning_rate": 1e-4,
        "warmup_proportion": 0.1
    }
}

relation_config ={

    "hyper_parameters": {
        "batch_size": 10,
        "epochs": 20,
        "learning_rate": 3e-5,
        "max_len": 512,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-5,
        "rewarm_epoch_num": 5,
        "T_mult": 1,
        "output_dir": "save_model",
        "train_path": 'data/duie_train.json',
        "valid_path": 'data/duie_dev.json',
        "weight_decay": 0.1,
        "model_path": "pretrained_model/RoBERTa_zh",
        "task_learning_rate": 3e-4,
        "warmup_proportion": 0.05
    }
}