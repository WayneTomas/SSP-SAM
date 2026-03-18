import torch
import torch.nn.functional as F
from torch import nn
import clip
from util import box_ops
import numpy as np
import math
from torch.nn.parameter import Parameter
from util.misc import get_world_size, is_dist_avail_and_initialized
from typing_extensions import Tuple

from ..text_model import PhraseAttention
from losses import dice_loss
from losses import BinaryFocalLoss
import sys
sys.path.append("..")
from segment_anything import sam_model_registry
sam_checkpoint = "pretrained_checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h_visual_prompt"
from ..vl_transformer import VisionLanguageEncoder
from ..transformer import VisualEncoder


class SSP_SAM(nn.Module):
    def __init__(self, clip_pretrained, args, device='cpu') -> None:
        """ Initializes the model."""
        super().__init__()
        # 是否进行预训练（在rec上预训练）
        self.is_pretrain = args.is_pretrain

        # freeze paras for SAM
        self.clip_embedding_dim = 512
        self.target_length = 1024
        image_size = 224
        self.vit_patch_size = int(image_size / 16)
        self.encoder, _ = clip.load(clip_pretrained, device=device)

        lstm_cfg=dict(
                        num_layers=1,
                        dropout=0.,
                        hidden_size=512,
                        bias=True,
                        bidirectional=True,
                        batch_first=True
                    )

        self.text_proj = nn.GRU(**lstm_cfg, input_size=512)
        self.clip_proj = nn.Linear(self.clip_embedding_dim, 512)

        self.phrase_attn = PhraseAttention(input_dim=512*2)
        self.mlp = nn.Linear(512, 256)
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        self.d_model = 256
        self.vl_transformer = VisionLanguageEncoder(
                                d_model=self.d_model,
                                dropout=0.,
                                nhead=8,
                                dim_feedforward=2048,
                                num_encoder_layers=6,
                                normalize_before=False,
                            )
        
        # with DETR
        # self.vl_transformer = VisualEncoder(d_model=self.d_model, num_encoder_layers=6, dropout=0.)
        # self.load_pretrained_weights('pretrained_checkpoints/detr-r50-unc.pth')

        self.query_token_num = 128
        num_total = self.vit_patch_size * self.vit_patch_size + self.query_token_num + 1
        self.vl_pos_embed = nn.Embedding(num_total, self.d_model)
        self.prompt_token = nn.Embedding(self.query_token_num, self.d_model)

        self.norm = nn.LayerNorm(self.d_model)
        self.norm_img = nn.LayerNorm(512)

        # bbox reggression aux
        self.bbox_embed = MLP(256, 256, 4, 3)
        self.reg_token = nn.Embedding(1, 256)

        # 
        self.img2text_attn = MULTIHEAD_ATTNS['MultiheadAttention'](**{'embed_dim': 512, 'num_heads': 8, 'dropout': 0.1})
        self.imgtext_proj = MLP(**{'input_dim': 512, 'hidden_dim': 512, 'output_dim': 512, 'num_layers': 1})
        self.img_proj = MLP(**{'input_dim': 512, 'hidden_dim': 512, 'output_dim': 512, 'num_layers': 1})
        self.tf_pow = 2.0
        self.tf_scale = Parameter(torch.Tensor([1.0]))
        self.tf_sigma = Parameter(torch.Tensor([0.5]))
        
        self.img2img_attn = MULTIHEAD_ATTNS['MHAttentionRPE'](**{'d_model': 256, 'h': 8, 'dropout': 0.1, 'pos_x_range': [-14, 14], 'pos_y_range': [-14, 14], 'pos_index_offset': 14})
        self.norm_text_cond_img = nn.LayerNorm(self.d_model)

        self.bbox_post = BBoxPostProcess()
        self.clip_img_proj = MLP(self.clip_embedding_dim,512,512,3)


    def load_pretrained_weights(self, weights_path):
        def load_weights(module, prefix, weights):
            module_keys = module.state_dict().keys()
            weights_keys = [k for k in weights.keys() if prefix in k]
            update_weights = dict()
            for k in module_keys:
                prefix_k = prefix+'.'+k
                if prefix_k in weights_keys:
                    update_weights[k] = weights[prefix_k]
                else:
                    print(f"Weights of {k} are not pre-loaded.")
            module.load_state_dict(update_weights, strict=False)

        weights = torch.load(weights_path, map_location='cpu')['model']
        # load_weights(self.backbone, prefix='backbone', weights=weights)
        load_weights(self.vl_transformer, prefix='transformer', weights=weights)
        # load_weights(self.input_proj, prefix='input_proj', weights=weights)
        print(f"Weights of DETR are pre-loaded.")

    def forward(self, image, image_mask, text, text_mask, sam_image, ori_size, target_dict):
        # get base image features
        bs = image.shape[0]
        image_features = self.encoder.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        class_embeddings, word_token_embedding = self.encoder.encode_text(text)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        # Apply feature surgery
        
        similarity_list = []
        for i in range(0, bs):
            similarity_list.append((image_features[i,:,:][None] @ (class_embeddings[i,:][None]).t()))
        similarity = torch.stack(similarity_list,dim=0).squeeze(1)
        similarity_map = clip.get_similarity_map(similarity[:, 1:, :])


        # adapt text attention
        clip_text_embed = self.clip_proj(word_token_embedding)
        y_word, h = self.text_proj(clip_text_embed)
        y_mask = torch.abs(text) == 0
        attn, weighted_emb = self.phrase_attn(context=y_word, embedded=clip_text_embed, mask=y_mask)
        
        # apply attn adaption of image and language
        select_image_features = image_features[:, 1:, :].view(-1, self.vit_patch_size, self.vit_patch_size, self.clip_embedding_dim) * similarity_map
        select_image_features = self.clip_img_proj(select_image_features)
        y_2d = weighted_emb.squeeze().unsqueeze(-1).unsqueeze(-1)

        # dis test
        # visual-linguistic verification
        img_query = select_image_features.view(bs, self.vit_patch_size * self.vit_patch_size, 512)

        text_info = self.img2text_attn(
                    query=img_query.permute(1,0,2), key=clip_text_embed.permute(1,0,2),
                    value=clip_text_embed.permute(1,0,2), key_padding_mask=y_mask)[0]

        text_embed = self.imgtext_proj(text_info).permute(1,0,2)
        img_embed = self.img_proj(img_query)
        verify_score = (F.normalize(img_embed, p=2, dim=-1) *
                        F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        verify_score = self.tf_scale * \
            torch.exp(- (1 - verify_score).pow(self.tf_pow)
                      / (2 * self.tf_sigma**2))
        fuse_img_feat = self.norm_img(img_query) * verify_score

        # ###########################################################################
        x_multi_modal = torch.tanh(fuse_img_feat.permute(0, 2, 1).contiguous()) * torch.tanh(y_2d.squeeze().unsqueeze(-1))
        x_multi_modal = x_multi_modal.permute(2, 0, 1).contiguous()
        x_multi_modal = self.mlp(x_multi_modal)

        # use transformer to fuse the multi-modal prompt
        bbox_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        bbox_mask = torch.zeros((bs, 1)).to(bbox_src.device).to(torch.bool)

        tgt_src = self.prompt_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, self.query_token_num)).to(tgt_src.device).to(torch.bool)

        x_multi_modal_mask = torch.zeros((bs, x_multi_modal.shape[0])).to(tgt_src.device).to(torch.bool)
        
        vl_src = torch.cat([bbox_src, tgt_src, x_multi_modal], dim=0)
        vl_mask = torch.cat([bbox_mask, tgt_mask, x_multi_modal_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # for vl transformer encoder
        output = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        # for vision transformer encoder
        # output, _, _ = self.vl_transformer(vl_src, vl_mask, vl_pos)  # (1+L+N)xBxC
        if self.norm is not None:
            hs = self.norm(output)
        
        bbox_token = hs[0]
        multi_prompt = x_multi_modal.permute(1, 0, 2)

        outputs_coord = self.bbox_embed(bbox_token).sigmoid()
        out = {'pred_boxes': outputs_coord.unsqueeze(1)}
        if self.is_pretrain:
            return out
        post_coord = self.bbox_post(out, target_dict)
        sam_type_pseudo_coord = []
        for i in range(bs):
            sam_type_pseudo_coord.append(self.apply_boxes_torch(post_coord[i], ori_size[i].detach().cpu().tolist()))
        sam_bbox = torch.cat(sam_type_pseudo_coord,dim=0)
        # use sam to get the masks
        batched_input = {"image": sam_image, "prompts": multi_prompt, "ori_size": ori_size, 'boxes': None}
        outputs, upscaled_embedding = self.sam(batched_input, multimask_output=False)

        out['pred_masks'] = outputs

        return out

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        # coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

def visualize_sam(masks: torch.Tensor, random_color: bool=False, cv2_img=None, idx=0, ori_size=None):
    # 压掉batch维度
    import matplotlib.pyplot as plt
    masks = torch.nn.functional.interpolate(masks[idx][None], (int(ori_size[idx][0]), int(ori_size[idx][1]))).squeeze(0)
    mask_np = masks.detach().cpu().numpy()
    
    vis = cv2_img.copy()
    # mask = mask_bin.astype('uint8')
    # mask = masks.squeeze()
    mask_bin = mask_np > 0
    mask = mask_bin.astype('uint8')
    vis[mask[0] > 0] = vis[mask[0] > 0] // 2 + np.array([153, 255, 255], dtype=np.uint8) // 2
    # vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
    plt.imsave("pred_mask_with_img.png", vis)

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask_np.shape[-2:]
    mask_image = mask_bin.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imsave("pred_mask.png", mask_image)


def visualize_sam_only(masks: torch.Tensor, random_color: bool=False, cv2_img=None, idx=0):
    # 压掉batch维度
    import matplotlib.pyplot as plt

    mask_np = masks[idx].detach().cpu().numpy()
    # vis = cv2_img.copy()
    # # mask = mask_bin.astype('uint8')
    # # mask = masks.squeeze()
    mask_bin = mask_np > 0
    # mask = mask_bin.astype('uint8')
    # vis[mask[0] > 0] = vis[mask[0] > 0] // 2 + np.array([153, 255, 255], dtype=np.uint8) // 2
    # # vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
    # plt.imsave("pred_mask_with_img.png", vis)

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask_np.shape[-2:]
    mask_image = mask_bin.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imsave("pred_mask.png", mask_image)


def avg_across_gpus(v, min=1):
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(v)
    return torch.clamp(v.float() / get_world_size(), min=min).item()


class VGCriterion(nn.Module):
    """ This class computes the loss for VLTVG."""
    def __init__(self, weight_dict, loss_loc, box_xyxy):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.weight_dict = weight_dict

        self.box_xyxy = box_xyxy

        self.loss_map = {'loss_boxes': self.loss_boxes}

        self.loss_loc = self.loss_map[loss_loc]

    def loss_boxes(self, outputs, target_boxes, num_pos):
        """Compute the losses related to the bounding boxes (the L1 regression loss and the GIoU loss)"""
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes'] # [B, #query, 4]
        target_boxes = target_boxes[:, None].expand_as(src_boxes)

        src_boxes = src_boxes.reshape(-1, 4) # [B*#query, 4]
        target_boxes = target_boxes.reshape(-1, 4) #[B*#query, 4]

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['l1'] = loss_bbox.sum() / num_pos

        if not self.box_xyxy:
            src_boxes = box_ops.box_cxcywh_to_xyxy(src_boxes)
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
        loss_giou = 1 - box_ops.box_pair_giou(src_boxes, target_boxes)
        losses['giou'] = (loss_giou[:, None]).sum() / num_pos
        return losses


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        """
        gt_boxes = targets['bbox']
        pred_boxes = outputs['pred_boxes']

        losses = {}
        B, Q, _ = pred_boxes.shape
        num_pos = avg_across_gpus(pred_boxes.new_tensor(B*Q))
        loss = self.loss_loc(outputs, gt_boxes, num_pos)
        losses.update(loss)

        # Apply the loss function to the outputs from all the stages
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                l_dict = self.loss_loc(aux_outputs, gt_boxes, num_pos)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class SegCriterion(nn.Module):
    """ This class computes the loss for VLTVG."""
    def __init__(self, weight_dict, loss_loc):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        self.weight_dict = weight_dict


        self.loss_map = {'loss_masks': self.loss_masks}

        self.loss_loc = self.loss_map[loss_loc]

        # self.dice_loss = DiceLoss(mode=BINARY_MODE)
        self.dice_loss = dice_loss
        # self.focal_loss = FocalLoss(mode=BINARY_MODE)
        # self.focal_loss = SoftBCEWithLogitsLoss()
        self.focal_loss = BinaryFocalLoss()
    
    def loss_masks(self, pred, gt):
        loss_1 = self.dice_loss(pred.squeeze(1), gt, gt.shape[0])
        loss_2 = self.focal_loss(pred, gt)
        losses = {}
        losses['dice'] = loss_1
        losses['focal'] = loss_2
        return losses

    def forward(self, pred, gt):
        """ This performs the loss computation.
        """
        losses = {}
        loss = self.loss_masks(pred, gt)
        losses.update(loss)

        return losses

class BBoxPostProcess(nn.Module):
    """ This module converts the model's output into the format we expect"""
    def __init__(self, box_xyxy=False):
        super().__init__()
        self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        """ Perform the computation"""
        rsz_sizes, ratios, orig_sizes = \
            target_dict['size'], target_dict['ratio'], target_dict['orig_size']
        dxdy = None if 'dxdy' not in target_dict else target_dict['dxdy']

        boxes = outputs['pred_boxes']

        assert len(boxes) == len(rsz_sizes)
        assert rsz_sizes.shape[1] == 2

        boxes = boxes.squeeze(1)

        # Convert to absolute coordinates in the original image
        if not self.bbox_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = rsz_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct
        if dxdy is not None:
            boxes = boxes - torch.cat([dxdy, dxdy], dim=1)
        boxes = boxes.clamp(min=0)
        ratio_h, ratio_w = ratios.unbind(1)
        boxes = boxes / torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=1)
        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            boxes = torch.min(boxes, torch.stack([orig_w, orig_h, orig_w, orig_h], dim=1))

        return boxes


class PostProcess(nn.Module):
    """ This module converts the model's output into the format we expect"""
    def __init__(self, box_xyxy=False):
        super().__init__()
        self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        """ Perform the computation"""
        rsz_sizes, ratios, orig_sizes = \
            target_dict['size'], target_dict['ratio'], target_dict['orig_size']
        dxdy = None if 'dxdy' not in target_dict else target_dict['dxdy']

        boxes = outputs['pred_boxes']

        assert len(boxes) == len(rsz_sizes)
        assert rsz_sizes.shape[1] == 2

        boxes = boxes.squeeze(1)

        # Convert to absolute coordinates in the original image
        if not self.bbox_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        img_h, img_w = rsz_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct
        if dxdy is not None:
            boxes = boxes - torch.cat([dxdy, dxdy], dim=1)
        boxes = boxes.clamp(min=0)
        ratio_h, ratio_w = ratios.unbind(1)
        boxes = boxes / torch.stack([ratio_w, ratio_h, ratio_w, ratio_h], dim=1)
        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            boxes = torch.min(boxes, torch.stack([orig_w, orig_h, orig_w, orig_h], dim=1))

        return boxes


class PostProcess_mask(nn.Module):
    """ This module converts the model's output into the format we expect"""
    def __init__(self, box_xyxy=False):
        super().__init__()
        # self.bbox_xyxy = box_xyxy

    @torch.no_grad()
    def forward(self, outputs, target_dict):
        """ Perform the computation"""
        rsz_sizes, ratios, orig_sizes = \
            target_dict['size'], target_dict['ratio'], target_dict['orig_size']

        masks = outputs

        post_masks = []
        if orig_sizes is not None:
            orig_h, orig_w = orig_sizes.unbind(1)
            post_masks.append()

        return post_masks


class MHAttentionRPE(nn.Module):
    ''' With relative position embedding '''

    def __init__(self, d_model, h, dropout=0.1, return_raw_attention=False,
                 pos_x_range=[-14, 14], pos_y_range=[-14, 14], pos_index_offset=14,
                 learnable_pos_embed=False):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.scaling = float(self.d_k) ** -0.5
        self.return_raw_attention = return_raw_attention

        self.in_proj_weight = Parameter(torch.Tensor(3 * d_model, d_model))
        self.in_proj_bias = Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn = None
        # self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout
        self._reset_parameters()

        self.learnable_pos_embed = learnable_pos_embed
        if learnable_pos_embed:
            self.pos_x = nn.Embedding(
                pos_x_range[1] - pos_x_range[0] + 1, d_model // 2)
            self.pos_y = nn.Embedding(
                pos_y_range[1] - pos_y_range[0] + 1, d_model // 2)
        else:
            pos_x, pos_y = position_embedding_sine(d_model // 2, normalize=True,
                                                   x_range=pos_x_range, y_range=pos_y_range)
            self.register_buffer('pos_x', pos_x)  # [x_range, C]
            self.register_buffer('pos_y', pos_y)  # [y_range, C]

        self.pos_index_offset = pos_index_offset

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None):
        tgt_len, bs, dim = query.size()
        src_len, _, dim = key.size()

        weight_q, bias_q = self.in_proj_weight[0:dim], self.in_proj_bias[0:dim]
        weight_k, bias_k = self.in_proj_weight[dim:dim *
                                               2], self.in_proj_bias[dim:dim*2]
        weight_v, bias_v = self.in_proj_weight[dim *
                                               2:], self.in_proj_bias[dim*2:]

        q = query.matmul(weight_q.t()) + bias_q
        k = key.matmul(weight_k.t()) + bias_k
        v = value.matmul(weight_v.t()) + bias_v

        # [bs*h, tgt_len, dim//h]
        q = q.view(tgt_len, bs * self.h, -1).transpose(0, 1)
        # [bs*h, dim//h, src_len], To calculate qTk (bmm)
        k = k.view(src_len, bs * self.h, -1).permute(1, 2, 0)
        v = v.view(src_len, bs * self.h, -1).transpose(0, 1)

        q = q * self.scaling
        attn_weights = torch.bmm(q, k)  # [bs*h, tgt_len, src_len]

        # compute the relative positions
        bs, HW = key_padding_mask.size()
        assert (HW == 196) and (HW == tgt_len)
        img_mask = ~key_padding_mask.view(bs, 14, 14)
        yy = img_mask.cumsum(1, dtype=torch.float32).view(
            bs, -1)  # [bs, HW],  1~20
        xx = img_mask.cumsum(2, dtype=torch.float32).view(
            bs, -1)  # [bs, HW],  1~20
        diff_yy = yy[:, :, None] - yy[:, None, :]  # [bs, HW, HW]
        diff_xx = xx[:, :, None] - xx[:, None, :]  # [bs, HW, HW]
        if self.learnable_pos_embed:
            k_posy = self.pos_y.weight.matmul(
                weight_k.t()[:dim//2])  # [x_range, dim]
            k_posx = self.pos_x.weight.matmul(
                weight_k.t()[dim//2:])  # [y_range, dim]
        else:
            k_posy = self.pos_y.matmul(weight_k.t()[:dim//2])  # [x_range, dim]
            k_posx = self.pos_x.matmul(weight_k.t()[dim//2:])  # [y_range, dim]
        k_posy = k_posy.view(-1, 1, self.h, dim//self.h).repeat(1, bs, 1, 1).\
            reshape(-1, bs * self.h, dim//self.h).permute(1,
                                                          2, 0)  # [bs*h, dim//h, y_range]
        k_posx = k_posx.view(-1, 1, self.h, dim//self.h).repeat(1, bs, 1, 1).\
            reshape(-1, bs * self.h, dim//self.h).permute(1,
                                                          2, 0)  # [bs*h, dim//h, x_range]
        posy_attn_weights = torch.bmm(q, k_posy).view(
            bs, self.h, HW, -1)  # [bs, h, HW, y_range]
        posx_attn_weights = torch.bmm(q, k_posx).view(
            bs, self.h, HW, -1)  # [bs, h, HW, x_range]
        diff_yy_idx = diff_yy[:, None].repeat(
            1, self.h, 1, 1) + self.pos_index_offset
        diff_xx_idx = diff_xx[:, None].repeat(
            1, self.h, 1, 1) + self.pos_index_offset

        posy_attn_weights = torch.gather(
            posy_attn_weights, -1, diff_yy_idx.long())  # [bs, h, HW, HW]
        posx_attn_weights = torch.gather(
            posx_attn_weights, -1, diff_xx_idx.long())  # [bs, h, HW, HW]
        pos_attn_weights = (posy_attn_weights +
                            posx_attn_weights).view(bs*self.h, HW, -1)
        attn_weights = attn_weights + pos_attn_weights

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(-1, self.h, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(
                    2),  # [bs, 1, 1, src_len]
                float('-inf')
            )
            attn_weights = attn_weights.view(-1, tgt_len, src_len)
        raw_attn_weights = attn_weights
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = F.dropout(
            attn_weights, p=self.dropout_p, training=self.training)
        attn_output = torch.bmm(attn_weights, v)
        self.attn = attn_weights

        attn_output = attn_output.transpose(
            0, 1).contiguous().view(tgt_len, bs, -1)
        attn_output = F.linear(
            attn_output, self.out_proj.weight, self.out_proj.bias)
        if self.return_raw_attention:
            return attn_output, raw_attn_weights
        return attn_output, attn_weights


def position_embedding_sine(num_pos_feats=64, temperature=10000, normalize=False, scale=None,
                            x_range=[-14, 14], y_range=[-14, 14], device=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi

    x_embed = torch.arange(x_range[0], x_range[1] + 1, device=device)
    y_embed = torch.arange(y_range[0], y_range[1] + 1, device=device)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1] + eps) * scale
        x_embed = x_embed / (x_embed[-1] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='trunc')) / num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(1)
    pos_y = torch.stack(
        (pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(1)
    return pos_x, pos_y



MULTIHEAD_ATTNS = {
    'MultiheadAttention': nn.MultiheadAttention,
    'MHAttentionRPE': MHAttentionRPE,
}


def build_vgmodel(args):
    device = torch.device(args.device)


    # for rec
    weight_dict_rec = {'loss_cls': 1, 'l1': args.bbox_loss_coef}
    weight_dict_rec['giou'] = args.giou_loss_coef

    # for res
    weight_dict = {'loss_cls': 1, 'dice': args.dice_loss_coef}
    weight_dict['focal'] = args.focal_loss_coef
    weight_dict.update(args.other_loss_coefs)
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    model = SSP_SAM(clip_pretrained=args.clip_pretrained, args=args, device=device)
    # criterion = VGCriterion(weight_dict=weight_dict, loss_loc=args.loss_loc, box_xyxy=args.box_xyxy)
    criterion_res = SegCriterion(weight_dict=weight_dict, loss_loc=args.loss_loc)
    criterion_res.to(device)

    criterion_rec = VGCriterion(weight_dict=weight_dict_rec, loss_loc=args.loss_loc_rec, box_xyxy=args.box_xyxy)
    criterion_rec.to(device)

    criterion_out = {"res":criterion_res,"rec":criterion_rec}

    postprocessor = PostProcess(args.box_xyxy)
    

    return model, criterion_out, postprocessor


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
