#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File : my_args.py
@Author : Yan Mingxue
@Software : PyCharm
"""

if_pi = False
# if_pi = True
# if_sigma = False
if_sigma = True

args = {
    'batch_size': 400,  # 256
    'pred_horizon': 20,
    'mhe_horizon': 40,
    # latent dimension
    'latent_dim': 13,  # lifted dimension
    'state_dim': 9,
    'output_dim': 3,
    'act_dim': 3,
    'seed': 7,
    # import data
    'import_saved_data': False,
    'continue_data_collection': False,  # if True: continue training

    'total_data_size': 2000,
    'max_ep_steps': 300,
    'test_steps': 10,
    'max_test_ep_steps': 100,
    'mhe_test_length': 20,
    'ABCD': 1,
    'val_frac': 0.2,
    'lr1': 0.01,
    'lr2': 0.001,
    'lr3': 0.01,
    'gamma': 0.6,
    # 'mix_x_u': 3,
    # 'if_mix': False,
    'num_epochs': 151,
    'weight_decay': 10,
    'extend_state': True,
    'restore': False,

    'if_pi': if_pi,
    'reload_data': True,  # Use previous dataset
    # 'reload_data': False,  # Regenerate new dataset
    # 'if_sigma': True,
    'if_sigma': False,

    'plot_test': True,
    'optimize_step': 20,
    'act_expand': 1,  # do not expand the input

    'SAVE_TEST': "save_model/test.pt",
    'SAVE_TRAIN': "save_model/train.pt",
    'SAVE_VAL': "save_model/val.pt",
    'SAVE_SHIFT': "save_model/shift.pt",
    'SAVE_TEST_X': "save_model/test_x.pt",
    'SAVE_TEST_U': "save_model/test_u.pt",

    'MODEL_SAVE' : "save_model/pi/model_v1.pt" if if_pi else "save_model/nopi/model_v1.pt",
    # 'NOISE_SAVE' : "save_model/pi/noise.pt" if if_pi else "save_model/nopi/noise.pt",
    'NOISE_SAVE' : "save_model/noise.pt",
    'SAVE_A1' : "save_model/pi/A1.pt"if if_pi else "save_model/nopi/A1.pt",
    'SAVE_B1' : "save_model/pi/B1.pt"if if_pi else "save_model/nopi/B1.pt",
    'SAVE_C1' : "save_model/pi/C1.pt"if if_pi else "save_model/nopi/C1.pt",
    'SAVE_OPTI1': "save_model/pi/opti1_v1.pt"if if_pi else "save_model/nopi/opti1_v1.pt",
    'SAVE_OPTI2': "save_model/pi/opti2_v1.pt"if if_pi else "save_model/nopi/opti2_v1.pt",
}

# args['MODEL_SAVE'] = "save_model/pi/model_v1.pt" if args['if_pi'] else "save_model/nopi/model_v1.pt",
# args['NOISE_SAVE'] = "save_model/nopi/noise.pt",
# args['SAVE_A1'] = "save_model/pi/A1.pt"if args['if_pi'] else "save_model/nopi/A1.pt",
# args['SAVE_B1'] = "save_model/pi/B1.pt"if args['if_pi'] else "save_model/nopi/B1.pt",
# args['SAVE_C1'] = "save_model/pi/C1.pt"if args['if_pi'] else "save_model/nopi/C1.pt",
# args['SAVE_OPTI1'] = "save_model/pi/opti1_v1.pt"if args['if_pi'] else "save_model/nopi/opti1_v1.pt",
# args['SAVE_OPTI2'] = "save_model/pi/opti2_v1.pt"if args['if_pi'] else "save_model/nopi/opti2_v1.pt",
