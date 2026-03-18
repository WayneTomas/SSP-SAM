# model report in main table
from .ssp_sam_336 import build_vgmodel
# from .ssp_sam_224 import build_vgmodel

# 
# ablation study for model archtecture # pretrain
# from .ablation.ssp_sam_224_only_LAD import build_vgmodel
# from .ablation.ssp_sam_224_only_Q_token import build_vgmodel
# from .ablation.ssp_sam_224_only_VAD import build_vgmodel
# from .ablation.ssp_sam_224_only_LAD_VAD import build_vgmodel
# from .ablation.ssp_sam_224_only_VAD_Q_token import build_vgmodel
# from .ablation.ssp_sam_224_only_LAD_Q_token import build_vgmodel
# from .ablation.ssp_sam_w_decoder import build_vgmodel
#
# ablation study for single task # no_pre, pre
# from .ablation.ssp_sam_224_single_task import build_vgmodel

# ablation study for tcsvt w/o gaussian
# from .ablation.ssp_sam_224_wo_gaussian import build_vgmodel

# ablation study for tcsvt siglip2
# from .ablation.ssp_sam_224_siglip2 import build_vgmodel

# ablation study for tcsvt eva-clip
# from .ablation.ssp_sam_224_eva_clip import build_vgmodel

# ablation study for num layer
# from .ablation.ssp_sam_224_layer_ablation import build_vgmodel


def build_model(args):
    return build_vgmodel(args)
