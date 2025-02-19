from net import Oryon

import mae.models_lib as models_lib
from mae.util import misc
from torch import nn            
        
class Oryon_ours(Oryon):
    
    def __init__(self, args, device):
        super().__init__(args, device)

        self.vlm = None
        self.fusion = None

        self.model = models_lib.__dict__["mae_croco"](img_size=(224,224))
        misc.dynamic_load_pretrain(self.model, "/home/robot/Repositories_chaoran/MPI/checkpoints/checkpoint-399-bridge-co3d.pth", interpolate=True)
        self.projector1 = nn.Linear(512, 128)
        self.projector2 = nn.Linear(512, 128)


    def forward(self, xs: dict):

        guid_a = self.get_guidance_embeds(xs['anchor']['rgb'])        
        guid_q = self.get_guidance_embeds(xs['query']['rgb'])        

        lang = [prompt[18] for prompt in xs['prompt']]

        feats_a = self.model.forward_6d_full_mask(xs['anchor']['rgb'], lang)
        feats_q = self.model.forward_6d_full_mask(xs['query']['rgb'], lang)
        
        feats_a = self.projector1(feats_a)
        feats_q = self.projector2(feats_q)

        feats_a = feats_a.reshape(-1,14,14,128)
        feats_q = feats_q.reshape(-1,14,14,128)

        # interpolate 14 -> 24
        feats_q = feats_q.permute(0,3,1,2)
        feats_a = feats_a.permute(0,3,1,2)
        feats_a = nn.functional.interpolate(feats_a, size=(24,24), mode='bilinear', align_corners=False)
        feats_q = nn.functional.interpolate(feats_q, size=(24,24), mode='bilinear', align_corners=False)

        feats_a = feats_a.unsqueeze(2)
        feats_q = feats_q.unsqueeze(2)

        mask_a, featmap_a = self.decoder.forward(feats_a, guid_a) # feats_a: [B, 128, 1, 24, 24]
        mask_q, featmap_q = self.decoder.forward(feats_q, guid_q)
        
        assert featmap_a.shape[2:] == self.args.image_encoder.img_size

        return {
            'featmap_a' : featmap_a,
            'featmap_q' : featmap_q,
            'mask_a' : mask_a,
            'mask_q' : mask_q
        }

    def get_trainable_parameters(self) -> list:

        param_list = []
        param_list.extend(self.decoder.parameters())
        param_list.extend(self.model.decoder_embed.parameters())
        param_list.extend(self.model.dec_pos_embed)
        param_list.extend(self.model.dec_norm.parameters())
        param_list.extend(self.model.dec_blocks.parameters())
        param_list.extend(self.projector1.parameters())
        param_list.extend(self.projector2.parameters())

        return param_list

    def train(self, mode=True):
        
        self.training = mode
        self.model.train(mode)
        self.decoder.train(mode)
        self.projector1.train(mode)
        self.projector2.train(mode)
        
        return self


class Oryon_ours_encoder(Oryon):
    
    def __init__(self, args, device):
        super().__init__(args, device)

        self.model = models_lib.__dict__["mae_croco"](img_size=(224,224))
        misc.dynamic_load_pretrain(self.model, "/home/robot/Repositories_chaoran/MPI/checkpoints/checkpoint-399-bridge-co3d.pth", interpolate=True)
        self.projector1 = nn.Linear(768, 1024)
        self.projector2 = nn.Linear(768, 1024)


    def forward(self, xs: dict):

        guid_a = self.get_guidance_embeds(xs['anchor']['rgb'])        
        guid_q = self.get_guidance_embeds(xs['query']['rgb'])        

        prompt_emb = self.vlm.encode_prompt(xs['prompt'])

        visual_a, _, _ = self.model.forward_encoder(xs['anchor']['rgb'], mask_ratio=0.0)
        visual_q, _, _ = self.model.forward_encoder(xs['query']['rgb'], mask_ratio=0.0)
        
        visual_a = self.projector1(visual_a)
        visual_q = self.projector2(visual_q)

        visual_a = visual_a[:,1:,...]
        visual_q = visual_q[:,1:,...]

        visual_a = visual_a.reshape(-1,14,14,1024)
        visual_q = visual_q.reshape(-1,14,14,1024)

        # interpolate 14 -> 24
        visual_q = visual_q.permute(0,3,1,2)
        visual_a = visual_a.permute(0,3,1,2)
        visual_a = nn.functional.interpolate(visual_a, size=(24,24), mode='bilinear', align_corners=False)
        visual_q = nn.functional.interpolate(visual_q, size=(24,24), mode='bilinear', align_corners=False)

        prompt_emb = prompt_emb.unsqueeze(1)
        feats_a = self.fusion.forward(visual_a, prompt_emb, guid_a)
        feats_q = self.fusion.forward(visual_q, prompt_emb, guid_q)

        mask_a, featmap_a = self.decoder.forward(feats_a, guid_a) # feats_a: [B, 128, 1, 24, 24]
        mask_q, featmap_q = self.decoder.forward(feats_q, guid_q)
        
        assert featmap_a.shape[2:] == self.args.image_encoder.img_size

        return {
            'featmap_a' : featmap_a,
            'featmap_q' : featmap_q,
            'mask_a' : mask_a,
            'mask_q' : mask_q
        }

    def get_trainable_parameters(self) -> list:

        param_list = []
        param_list.extend(self.projector1.parameters())
        param_list.extend(self.projector2.parameters())
        param_list.extend(self.fusion.parameters())
        param_list.extend(self.decoder.parameters())

        return param_list

    def train(self, mode=True):
        
        self.training = mode
        self.vlm.train(mode)
        self.model.train(mode)
        self.decoder.train(mode)
        self.projector1.train(mode)
        self.projector2.train(mode)
        self.fusion.train(mode)
        
        return self