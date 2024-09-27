# 导入所需的库和模块
import numpy as np
import torch 
import torch.nn as nn
import timm 
import open_clip
from typing import Any, Dict, Optional
from types import SimpleNamespace
from collections import OrderedDict

# 定义一个命名空间 ModalityType，用于表示不同的模态类型
ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    TACTILE="tactile"
)

# 定义用于视觉模型的CLIP模型名称和预训练数据
CLIP_VISION_MODEL = "ViT-L-14"
CLIP_PRETRAIN_DATA = "datacomp_xl_s13b_b90k"

# 获取CLIP视觉模型的分词器
tokenizer = open_clip.get_tokenizer(CLIP_VISION_MODEL)

class TVL(nn.Module):    # 定义一个名为 TVL 的类，它继承自PyTorch的 nn.Module
    # TVL 类的构造函数，初始化模型的不同部分
    def __init__(
        self, active_modalities = [ModalityType.VISION, ModalityType.TACTILE, ModalityType.TEXT], 
        clip_vision_model=CLIP_VISION_MODEL, 
        clip_pretrain_data=CLIP_PRETRAIN_DATA, 
        tactile_model="vit_tiny_patch16_224", 
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        common_latent_dim: int = None, # for imagebind this is set to 1024, and last layer has width 1280 (ViT-H-14)
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super(TVL, self).__init__()    # 调用父类的构造函数
        assert len(active_modalities) > 1, "At least two modalities must be active"    # 确保至少有两个模态是激活的 
        self.active_modalities = active_modalities    # 存储激活的模态
        # 创建CLIP模型和预处理变换
        self.clip, _, self.vision_preprocess = open_clip.create_model_and_transforms(clip_vision_model, pretrained=clip_pretrain_data)
        self.tokenizer = open_clip.get_tokenizer(clip_vision_model)    # 获取CLIP模型的分词器

        # 根据是否指定 common_latent_dim 来设置类别数
        if common_latent_dim is not None: 
            # then we will put all the modality head self.modality_head
            assert common_latent_dim > 0, "common_latent_dim must be positive"
            num_classes = 0 
        else:
            # we merge the modality head into the model
            num_classes = self.clip.transformer.width

        # 使用 timm 创建触觉模型
        self.tactile_encoder = timm.create_model(tactile_model, pretrained=False, num_classes=num_classes, global_pool="avg", drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)    # 初始化logit scale参数
        # 初始化logit bias参数，如果没有指定则为None
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

        # 初始化模态头字典和公共潜在维度
        modality_heads = {}
        self.common_latent_dim = common_latent_dim
        # 如果指定了公共潜在维度，则为每个激活的模态创建一个线性层
        if common_latent_dim is not None:
            for modality in self.active_modalities:
                if modality == ModalityType.TACTILE:
                    modality_heads[modality] = nn.Linear(self.tactile_encoder.num_features, common_latent_dim, bias=False)
                else:
                    modality_heads[modality] = nn.Linear(self.clip.transformer.width, common_latent_dim, bias=False)
        self.modality_heads = nn.ModuleDict(modality_heads)

        # 默认情况下冻结视觉和文本模型的参数
        # by default, we freeze openclip 
        self.freeze_vision()
        self.freeze_text()

        # 如果视觉或文本模态未激活，则移除相应的模块
        if ModalityType.VISION not in self.active_modalities:
            # we remove the clip.visual module
            del self.clip.visual
        if ModalityType.TEXT not in self.active_modalities:
            # we remove the clip.transformer module
            del self.clip.transformer
        
        # we clear torch cache 
        torch.cuda.empty_cache()    # 清空PyTorch的CUDA缓存

    # 冻结CLIP模型的所有参数
    def freeze_openclip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    # 冻结视觉模型的所有参数
    def freeze_vision(self):
        for param in self.clip.visual.parameters():
            param.requires_grad = False

    # 冻结触觉模型的所有参数
    def freeze_tactile(self):
        for param in self.tactile_encoder.parameters():
            param.requires_grad = False

    # 冻结文本模型的所有参数
    def freeze_text(self):
        for param in self.clip.transformer.parameters():
            param.requires_grad = False

    # 重写 state_dict 方法，移除CLIP相关的权重，只保存触觉编码器的权重
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(TVL, self).state_dict(destination, prefix, keep_vars)
        # we remove all clip related weights and only save the tactile encoder
        new_state_dict = OrderedDict()
        for k in state_dict:
            if "clip" not in k:
                new_state_dict[k] = state_dict[k]
        del state_dict
        return new_state_dict

    # 定义 forward 方法，它接受一个包含不同模态输入的字典
    def forward(self, input_dict : dict):
        # dictionary should have keys: vision, tactile, text
        # vision: (batch, 3, 224, 224)
        # tactile: (batch, 3, 224, 224)
        # text: (batch, 77)
        out_dict = {}
        # 如果输入字典包含视觉模态，则编码视觉特征
        if ModalityType.VISION in input_dict.keys():
            with torch.no_grad():
                vision_features = self.clip.encode_image(input_dict[ModalityType.VISION], normalize=True)
            out_dict[ModalityType.VISION] = vision_features
        # 如果输入字典包含触觉模态，则编码触觉特征并进行归一化
        if ModalityType.TACTILE in input_dict.keys():
            tactile_features = self.tactile_encoder(input_dict[ModalityType.TACTILE])
            # normalize tactile_features 
            tactile_features = tactile_features / torch.norm(tactile_features, dim=1, keepdim=True)
            out_dict[ModalityType.TACTILE] = tactile_features
        # 如果输入字典包含文本模态，则编码文本特征
        if ModalityType.TEXT in input_dict.keys():
            with torch.no_grad():
                text_features = self.clip.encode_text(input_dict[ModalityType.TEXT], normalize=True)
            out_dict[ModalityType.TEXT] = text_features
        # 返回输出字典，包括logit scale和logit bias（如果存在）
        out_dict["logit_scale"] = self.logit_scale.exp()
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        return out_dict
