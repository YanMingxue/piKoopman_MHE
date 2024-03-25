args = {
    'batch_size': 100,
    'pred_horizon': 20,
    'ctrl_horizon': 10,
    # latent dimension
    'latent_dim': 13,  # Koopman 变换后的向量数量
    # import data
    'import_saved_data': False,
    'continue_data_collection': False,  # 是否连续训练
    'total_data_size': 500,
    'max_ep_steps': 300,
    'test_steps': 30,
    'ABCD': 1,
    'val_frac': 0.2,
    'lr1': 0.01,
    'lr2': 0.001,
    'lr3': 0.01,
    'gamma': 0.8,
    'mix_x_u': 3,
    'if_mix': False,
    # 'num_epochs' : 80,
    'num_epochs': 121,
    'weight_decay': 10,

    'extend_state': True,

    'restore': False,

    'if_pi': True,
    # 'if_pi' : False,
    'reload_data': True,
    # 'reload_data': False,

    'plot_test': True,
    'optimize_step': 20,
    'act_expand': 1,  # 输入拓展的维数

    'MODEL_SAVE': "save_model/model_v1.pt",
    'SAVE_OPTI1': "save_model/opti1_v1.pt",
    'SAVE_OPTI2': "save_model/opti2_v1.pt",
    'SAVE_A1': "save_model/A1.pt",
    'SAVE_B1': "save_model/B1.pt",
    'SAVE_C1': "save_model/C1.pt",
    'SAVE_TEST': "save_model/test.pt",
    'SAVE_TRAIN': "save_model/train.pt",
    'SAVE_VAL': "save_model/val.pt",
    'SAVE_SHIFT': "save_model/shift.pt",
    'SAVE_TEST_X': "save_model/test_x.pt",
    'SAVE_TEST_U': "save_model/test_u.pt",

}
