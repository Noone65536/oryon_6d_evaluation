"""
Choose which model to load
"""
from models_mae_robot_lang import MAERobotLang
from models_croco import MAERobotLangCroco
from functools import partial
import torch.nn as nn


def mae_vit_base_patch16_rl(**kwargs):
    model = MAERobotLang(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_croco(**kwargs):
    model = MAERobotLangCroco(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# models
  # two state mae without language
mae_robot_lang = mae_vit_base_patch16_rl
mae_croco = mae_vit_base_patch16_croco
