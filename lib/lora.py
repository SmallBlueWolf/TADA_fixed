import torch

from .lora_unet import UNet2DConditionModel     
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import einops

def get_lora_unet(opt, guidance, device):
    if not opt.v_pred:
        _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(device)
    else:
        _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(device)
    _unet.requires_grad_(False)
    lora_attn_procs = {}
    for name in _unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = _unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = _unet.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
    _unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(_unet.attn_processors)

    text_input = guidance.tokenizer(opt.text, padding='max_length', max_length=guidance.tokenizer.model_max_length, truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_embeddings = guidance.text_encoder(text_input.input_ids.to(guidance.device))[0]

    class LoraUnet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unet = _unet
            self.sample_size = 64
            self.in_channels = 4
            self.device = device
            self.dtype = torch.float32
            self.text_embeddings = text_embeddings
        def forward(self,x,t,c=None,shading="albedo"):
            textemb = einops.repeat(self.text_embeddings, '1 L D -> B L D', B=x.shape[0]).to(device)
            return self.unet(x,t,encoder_hidden_states=textemb,c=c,shading=shading)
    return _unet, lora_layers, LoraUnet().to(device)    