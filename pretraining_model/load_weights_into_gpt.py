import torch
from gpt_model.model import GPT
import numpy as np

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt: GPT, params: dict):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].att.query.weight = assign(
            gpt.transformer_blocks[b].att.query.weight, q_w.T)
        gpt.transformer_blocks[b].att.key.weight = assign(
            gpt.transformer_blocks[b].att.key.weight, k_w.T)
        gpt.transformer_blocks[b].att.value.weight = assign(
            gpt.transformer_blocks[b].att.value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].att.query.bias = assign(
            gpt.transformer_blocks[b].att.query.bias, q_b)
        gpt.transformer_blocks[b].att.key.bias = assign(
            gpt.transformer_blocks[b].att.key.bias, k_b)
        gpt.transformer_blocks[b].att.value.bias = assign(
            gpt.transformer_blocks[b].att.value.bias, v_b)

        gpt.transformer_blocks[b].att.output.weight = assign(
            gpt.transformer_blocks[b].att.output.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].att.output.bias = assign(
            gpt.transformer_blocks[b].att.output.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].ff.net[0].weight = assign(
            gpt.transformer_blocks[b].ff.net[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].ff.net[0].bias = assign(
            gpt.transformer_blocks[b].ff.net[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].ff.net[2].weight = assign(
            gpt.transformer_blocks[b].ff.net[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].ff.net[2].bias = assign(
            gpt.transformer_blocks[b].ff.net[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[b].ln1.scale = assign(
            gpt.transformer_blocks[b].ln1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].ln1.shift = assign(
            gpt.transformer_blocks[b].ln1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].ln2.scale = assign(
            gpt.transformer_blocks[b].ln2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].ln2.shift = assign(
            gpt.transformer_blocks[b].ln2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.ln_f.scale = assign(gpt.ln_f.scale, params["g"])
    gpt.ln_f.shift = assign(gpt.ln_f.shift, params["b"])
    gpt.head.weight = assign(gpt.head.weight, params["wte"])
    