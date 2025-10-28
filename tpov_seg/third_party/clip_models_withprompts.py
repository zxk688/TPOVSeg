import torch
import torch.nn as nn
from transformers import CLIPModel

#############################################
### CLIP Text Encoder Parameter-Efficient ###
#############################################
"""
接收原始CLIP文本模型作为基础
引入了prompt tuning机制,通过 deep_g 参数控制在多少层添加prompt
支持三种prompt添加方式:"replace", "accumulate", "accumulate_same"
"""
class CLIPTextModelForPromptTuning(nn.Module):
    def __init__(self, model: object, deep_g: int, deep_replace_method: str = "replace"):
        '''
        CLIP Text Encoder for PE
        model: CLIP Text Encoder
        deep_g: number of layers to append prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same" prompts to every layer
        '''
        super().__init__()
        self.model = model
        self.d_model = 512
        self.deep_g = deep_g
        self.deep_replace_method = deep_replace_method

    def forward(self, 
            text_tokens: torch.Tensor,          #经过分词器处理之后的文本序列，例如[49406,   320,  1125,   539,   320, 1929, 49407]
            attn_mask: torch.Tensor,            #注意力掩码，用于控制哪些 tokens 之间可以进行注意力计算，格式相同，由0，1构成
            g_prompt: torch.Tensor              #可学习的 prompt 参数   shape: [batch_size, n_deep, n_prompts, d_model]
        ):
        '''
        text_tokens: [batch_size, n_tokens]
        attn_mask: [batch_size, n_tokens]
        g_prompt: [text_tokens, n_deep, n_prompts, d_model]
        '''
        bs = g_prompt.size(0)

        g_prompt = g_prompt.to(text_tokens.device)
        
        x = self.model.embeddings.token_embedding(text_tokens)
        g = g_prompt[:, 0]                      #在n_deep 维度上提取第一个切片                                     
        L_g = g.size(1)                         #获取n_prompts的维度的大小
        

        x = torch.cat([x[:,0:1,:], g, x[:,1:,:]], dim=1)            # 在原始token嵌入的基础上，在[CLS]token后，正常token之前，插入prompt
        x = x + self.model.embeddings.position_embedding(torch.arange(x.size(1), device=attn_mask.device).unsqueeze(0))

        for i,l in enumerate(self.model.encoder.layers):
            
            if i > 0:
                if i < self.deep_g:
                    if self.deep_replace_method == "replace":
                        g = g_prompt[:, i]
                    elif self.deep_replace_method == "accumulate":
                        previous_g_out = x[:,1:(L_g+1),:]
                        g = torch.cat([previous_g_out, g_prompt[:, i]], dim=1)
                    elif self.deep_replace_method == "accumulate_same":
                        g = torch.cat([g, g_prompt[:, i]], dim=1)
                    x = torch.cat([x[:,0:1,:], g, x[:,(L_g+1):,:]], dim=1)
                    L_g = g.size(1)

            attn_mask_ = torch.cat([torch.ones(bs, L_g, device=attn_mask.device), attn_mask], dim=-1)
                    
            res = x
            x = l.layer_norm1(x)

            q = l.self_attn.q_proj(x) * 0.125
            k = l.self_attn.k_proj(x)
            v = l.self_attn.v_proj(x)

            extended_attn_mask = (attn_mask_.unsqueeze(1).unsqueeze(1) == 0).float()
            extended_attn_mask[extended_attn_mask==1] = torch.finfo(x.dtype).min

            q = q.view(x.size(0), x.size(1), 8, -1).transpose(1,2)
            k = k.view(x.size(0), x.size(1), 8, -1).transpose(1,2)
            v = v.view(x.size(0), x.size(1), 8, -1).transpose(1,2)
            w = q @ k.transpose(-1,-2) 
            c_mask = self.model._build_causal_attention_mask(x.size(0), x.size(1), x.dtype).float().to(attn_mask.device)
            w = w + c_mask + extended_attn_mask
            w = w.softmax(dim=-1)
            v = (w @ v).transpose(1,2).contiguous().view(x.size(0), x.size(1), -1)
            x = l.self_attn.out_proj(v)
            x = res + x

            res = x
            x = l.layer_norm2(x)
            x = l.mlp(x)
            
            x = res + x

        x = self.model.final_layer_norm(x)

        index = text_tokens.argmax(dim=-1) + L_g
        return x[torch.arange(x.size(0)), index]
    
###############################################
### CLIP Vision Encoder Parameter-Efficient ###
###############################################
    
class CLIPVisionModelForPromptTuning(nn.Module):
    def __init__(self, 
            model: object, 
            deep_g: int, 
            deep_replace_method: str = "replace"
        ):
        '''
        CLIP Vision Encoder for PE
        model: CLIP Vision Encoder
        deep_g: number of layers to append prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same" prompts to every layer
        '''
        super().__init__()
        self.model = model
        self.d_model = 768
        self.deep_g = deep_g
        self.deep_replace_method = deep_replace_method

    def forward(self, 
            image: torch.Tensor, 
            g_prompt: torch.Tensor
        ):
        '''
        image: [batch_size, 3, 224, 224]
        g_prompt: [batch_size, n_deep, n_prompts, d_model]
        '''
        x = self.model.embeddings(image)

        g = g_prompt[:, 0]

        x = torch.cat([x, g], dim=1)
        x = self.model.pre_layrnorm(x)
        L_g = g.size(1)
        
        for i,l in enumerate(self.model.encoder.layers):

            if i > 0:
                if i < self.deep_g:
                    if self.deep_replace_method == "replace":
                        g = g_prompt[:, i]
                    elif self.deep_replace_method == "accumulate":
                        previous_g_out = x[:,-L_g:,:]
                        g = torch.cat([previous_g_out, g_prompt[:, i]], dim=1)
                    elif self.deep_replace_method == "accumulate_same":
                        g = torch.cat([g, g_prompt[:, i]], dim=1)
                    x = torch.cat([x[:, :-L_g, :], g], dim=1)
                    L_g = g.size(1)

            res = x
            x = l.layer_norm1(x)

            q = l.self_attn.q_proj(x) * 0.125
            k = l.self_attn.k_proj(x)
            v = l.self_attn.v_proj(x)

            q = q.view(x.size(0), x.size(1), 12, -1).transpose(1,2)
            k = k.view(x.size(0), x.size(1), 12, -1).transpose(1,2)
            v = v.view(x.size(0), x.size(1), 12, -1).transpose(1,2)
            w = q @ k.transpose(-1,-2)
            w = w.softmax(dim=-1)
            v = (w @ v).transpose(1,2).contiguous().view(x.size(0), x.size(1), -1)
            x = l.self_attn.out_proj(v)
            x = res + x

            res = x
            x = l.layer_norm2(x)
            x = l.mlp(x)
            
            x = res + x

        return self.model.post_layernorm(x[:,0,:])
    
################################
### CLIP Parameter-Efficient ###
################################

class CLIPParameterEfficient(nn.Module):
    def __init__(self, 
            L_g: int = 2, 
            deep_g: int = 3, 
            text_deep_replace_method: str = "replace",
            vision_deep_replace_method: str = "replace"
        ):
        '''
        CLIP Parameter-Efficient
        L_g: number of g-prompts
        deep_g: number of layers to attach g-prompts
        deep_replace_method: "replace", "accumulate", or "accumulate_same" prompts to every layer
        '''
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")   #加载完整的clip模型权重

        ### Text Encoder ###
        self.text_model = CLIPTextModelForPromptTuning(
            model = self.clip_model.text_model, 
            deep_g = deep_g, 
            deep_replace_method = text_deep_replace_method
        )

        ### Vision Encoder ###
        self.vision_model = CLIPVisionModelForPromptTuning(
            model = self.clip_model.vision_model, 
            deep_g = deep_g, 
            deep_replace_method = vision_deep_replace_method
        )

        self.image_proj = self.clip_model.visual_projection
        self.text_proj = self.clip_model.text_projection
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        
        self.prompt_proj = nn.Linear(self.text_model.d_model, self.vision_model.d_model)
        self.g_values = nn.Parameter(torch.zeros(deep_g, L_g, self.text_model.d_model))
        
        nn.init.xavier_uniform_(self.g_values.data)

        self.L_g = L_g
        self.deep_g = deep_g
        
    def forward(
            self, 
            image: torch.Tensor, 
            text_tokens: torch.Tensor, 
            attn_mask: torch.Tensor,
            device = "cuda"
        ):
        '''
        image: [batch_size, 3, 224, 224]
        text_tokens: [n_classes, max_length]
        attn_mask: [n_classes, max_length]
        '''
        batch_size = image.shape[0]

        text_g_prompt = self.g_values.repeat(text_tokens.size(0), 1, 1, 1).to(device)
        vision_g_prompt = self.prompt_proj(self.g_values.repeat(batch_size, 1, 1, 1))

        text_out = self.text_model(text_tokens, attn_mask, text_g_prompt)
        img_out = self.vision_model(image, vision_g_prompt) 
                 
        # Project to common dimensional space
        text_proj = self.text_proj(text_out)
        img_proj = self.image_proj(img_out)
        # Normalize
        text_embed = text_proj / text_proj.norm(dim=-1, keepdim=True)
        img_embed = img_proj / img_proj.norm(dim=-1, keepdim=True)
        sim = 100 * img_embed @ text_embed.T
                
        return sim
