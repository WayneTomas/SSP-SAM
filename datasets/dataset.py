import json
import os
import os.path as osp
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from torch.utils.data import Dataset

from .utils import convert_examples_to_features_clip, read_examples

from .transforms import PIL_TRANSFORMS
from segment_anything import sam_model_registry
sam_checkpoint = "pretrained_checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam_embedding_folder_opt = {
    # 1. you can change the path to save the SAM image embeddings, 
    # which can speed up the training process after the first epoch. 
    # 2. If you don't want to save the SAM image embeddings, 
    # you can set it to None, but it will slow down the training process.
    # 3. you can save the SAM image embeddings for all datasets in the same folder, 
    # or you can save them in different folders for different datasets.
    'unc': 'data_seg/embeddings/unc',
    'unc+': 'data_seg/embeddings/unc',
    'gref': 'data_seg/embeddings/unc',
    'gref_umd': 'data_seg/embeddings/unc',
    # 'unc+': 'data_seg/embeddings/unc+',
    # 'gref': 'data_seg/embeddings/gref',
    # 'gref_umd': 'data_seg/embeddings/gref',

    # merge has no SAM embedding because it is only used for pre-training
    'merge': None,
    'grefcoco': 'data_seg/embeddings/unc',
    'referit': 'data_seg/embeddings/referit',
    # for zero-shot
    'phrase_cut': 'data_seg/embeddings/phrase_cut',
}

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from clip.clip import tokenize



# Meta Information
SUPPORTED_DATASETS = {
    'referit': {'splits': ('train', 'val', 'trainval', 'test')},
    'unc': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB','testC'),
        'params': {'dataset': 'refcoco', 'split_by': 'unc'}
    },
    'unc+': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
    },
    'gref': {
        'splits': ('train', 'val'),
        'params': {'dataset': 'refcocog', 'split_by': 'google'}
    },
    'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
    },
    'flickr': {
        'splits': ('train', 'val', 'test')
        },
    'ref_reasoning': {
        'splits': ('train', 'val', 'test'),},
    'merge': {
        'splits': ('train', 'val'),
    },
    'grefcoco': {
        'splits': ('train', 'val', 'testA', 'testB'),
        'params': {'dataset': 'grefcoco', 'split_by': 'unc'}
    },
    'phrase_cut': {
        'splits': ('test'),
    },
}


class VGDataset(Dataset):
    def __init__(self, data_root, split_root='data', dataset='referit', transforms=[],
                 debug=False, test=False, split='train', max_query_len=128,
                 cache_images=False):
        super(VGDataset, self).__init__()

        self.device = 'cpu'
        self.long_tail_thd = 20
        self.long_tail = False
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(self.device)
        self.sam.eval()

        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.test = test
        self.transforms = []
        self.other_transforms = []
        self.mask_transforms = []
        self.ann_path = {
            "unc": "data_seg/anns/refcoco.json",
            "unc+": "data_seg/anns/refcoco+.json",
            "gref": "data_seg/anns/refcocog_google.json", #指的是refcocog-google split
            "gref_umd": "data_seg/anns/refcocog_umd.json", #指的是refcocog-umd split
            "referit": "data_seg/anns/refclef.json",
            "merge": "data_seg/anns/merge.json",
            "grefcoco": "data_seg/anns/grefcoco.json",
            "phrase_cut": "data_seg/anns/phrase_cut.json",
            }
        self.mask_path = {
            "unc": "data_seg/masks/refcoco",
            "unc+": "data_seg/masks/refcoco+",
            "gref": "data_seg/masks/refcocog_google",
            "gref_umd": "data_seg/masks/refcocog_umd",
            "referit": "data_seg/masks/refclef",
            "grefcoco": "data_seg/masks/grefcoco",
            "phrase_cut": "data_seg/masks/phrase_cut_mask",
            }
        self.sam_embedding_folder = sam_embedding_folder_opt[self.dataset]
        self.getitem = self.getitem__PIL
        self.read_image = self.read_image_from_path_PIL
        for t in transforms:
            _args = t.copy()
            self.transforms.append(PIL_TRANSFORMS[_args.pop('type')](**_args))
        # #############################################
        # written by wayne
        self.clip_transforms = Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.other_transforms.append(PIL_TRANSFORMS['SAMResize']())
        self.mask_transforms.append(PIL_TRANSFORMS['SAMMaskResize']())


        self.debug = debug

        self.query_len = max_query_len
        self.tokenizer = tokenize

        # setting datasource
        self.is_pretrain = False
        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit', 'images')
            self.im_dir = osp.join(self.dataset_root)
            stat_refs_list=json.load(open(self.ann_path[dataset], 'r'))
        elif self.dataset == 'ref_reasoning':
            self.dataset_root = osp.join(self.data_root, 'GQA')
            self.im_dir = osp.join(self.dataset_root, 'images')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        elif self.dataset in ['unc', 'unc+', 'gref', 'gref_umd']:  # refer coco etc.
            self.dataset_root = osp.join(self.data_root)
            self.im_dir = osp.join(self.dataset_root, 'coco', 'train2014')
            stat_refs_list=json.load(open(self.ann_path[dataset], 'r'))
        elif self.dataset == 'merge':
            self.im_dir_coco = osp.join(self.dataset_root, 'coco', 'train2014')
            self.img_dir_vg = osp.join(self.data_root, 'VG')
            self.img_dir_flickr = osp.join(self.data_root, 'flickr')
            self.img_dir_referit = osp.join(self.data_root, 'referit', 'images')
            stat_refs_list = json.load(open(self.ann_path['merge'], 'r'))
            self.is_pretrain = True
        elif self.dataset == 'phrase_cut':
            self.img_dir_vg = osp.join(self.data_root, 'vg')
            stat_refs_list=json.load(open(self.ann_path[dataset], 'r'))
        elif self.dataset in ['grefcoco']:  # refer coco etc.
            self.dataset_root = osp.join(self.data_root)
            self.im_dir = osp.join(self.dataset_root, 'coco', 'train2014')
            stat_refs_list=json.load(open(self.ann_path[dataset], 'r'))

        # dataset_split_root = osp.join(self.split_root, self.dataset)
        valid_splits = SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        self.refs_anno=[]
        
        splits = [split]
        for split in splits:
            self.refs_anno+= stat_refs_list[split]


        refs = []
        self.bboxs = {}
        
        if self.long_tail or self.is_pretrain:
            new_refs_anno = []

        pretrain_bbox_idx = 0
        self.task_id = {}
        self.task_i = {}
        for split in splits:
            for ann in stat_refs_list[split]:
                if not self.is_pretrain:
                    if self.dataset != 'phrase_cut':
                        self.bboxs[ann['mask_id']] = ann['bbox']
                        for ref in ann['refs']:
                            refs.append(ref)
                    else:
                        self.bboxs[pretrain_bbox_idx] = ann['bbox']
                        self.task_id[pretrain_bbox_idx] = ann['task_id']
                        self.task_i[pretrain_bbox_idx] = ann['task_i']
                        pretrain_bbox_idx += 1
                        for ref in ann['refs']:
                            refs.append(ref)
                else:
                    for ref in ann['refs']:
                        if len(ref.split(' ')) <= 40:
                            refs.append(ref)
                            new_refs_anno.append(ann)
                            self.bboxs[pretrain_bbox_idx] = ann['bbox']
                            pretrain_bbox_idx += 1
                
            if self.is_pretrain:
                self.refs_anno = new_refs_anno

        self.convert_bbox = {}
        # if not (self.dataset == 'referit' or self.dataset == 'flickr'):  # for refcoco, etc (merge数据里的flickr部分的bbox也转化为xywh的形式了)
        # 按照simrec处理的bbox是xywh格式
        if self.dataset in ['unc','unc+','gref', 'gref_umd', 'referit']:
            # xywh to xyxy
            for bbox_id, bbox in self.bboxs.items():
                bbox = np.array(bbox, dtype=np.float32)
                bbox[2:] += bbox[:2]
                self.convert_bbox[bbox_id] = bbox
        elif self.dataset == 'grefcoco':
            for bbox_id, bbox in self.bboxs.items():  # grefcoco(GRES)在处理的时候由于merge box，已经是xyxy了
                bbox = np.array(bbox, dtype=np.float32)
                self.convert_bbox[bbox_id] = bbox
        else:
            for bbox_id, bbox in self.bboxs.items():  # for flickr
                bbox = np.array(bbox, dtype=np.float32)
                self.convert_bbox[bbox_id] = bbox

    def __len__(self):
        return len(self.refs_anno)

    def image_path(self, idx):  # notice: db index is the actual index of data.
        return osp.join(self.im_dir, self.img_names[idx])

    def annotation_box(self, idx):
        return self.convert_bbox[idx].copy()

    def phrase(self, idx):
        return self.phrases[idx]

    def cache(self, idx):
        self.images_cached[idx] = self.read_image_orig_func(idx)

    def read_image_from_path_PIL(self, idx):
        image_path = self.image_path(idx)
        pil_image = Image.open(image_path).convert('RGB')
        return pil_image

    def read_image_from_cache(self, idx):
        image = self.images_cached[idx]
        return image

    def __getitem__(self, idx):
        return self.getitem(idx)


    def getitem__PIL(self, idx):
        # #######################################
        # written by wayne
        phrase = self.load_refs(idx)
        if self.dataset != 'phrase_cut':
            image,mask_iter,bbox, img_path= self.load_img_feats(idx)
        else:
            image,mask_iter,bbox,task_id, img_path= self.load_img_feats(idx)
        #########################################
        # reading images
        orig_image = image
        bbox = torch.tensor(bbox, dtype=torch.float32).squeeze()
        phrase = phrase.lower()
        orig_phrase = phrase

        target = {}
        target['phrase'] = phrase
        target['ori_size'] = torch.tensor([orig_image.height, orig_image.width], dtype=torch.int32)
        target['bbox'] = bbox
        
        if self.test or self.debug:
            target['orig_bbox'] = bbox.clone()

        # #####################################
        # written by wayne
        # transform for segment anything
        if not self.is_pretrain:
            for transform in self.other_transforms:
                image_sam = transform(np.array(image))
            
            sam_embedding_name = os.path.basename(img_path)
            sam_embedding_name = sam_embedding_name.replace('.jpg', '.npy')
            os.makedirs(self.sam_embedding_folder, exist_ok=True)
            sam_embedding_save_path = os.path.join(self.sam_embedding_folder, sam_embedding_name)

            if not os.path.exists(sam_embedding_save_path):
                with torch.no_grad():
                    image_sam_embedding = self.sam.image_encoder(image_sam.unsqueeze(0).to(self.device))
                np.save(sam_embedding_save_path, image_sam_embedding.squeeze().detach().cpu().numpy())
                image_sam = image_sam_embedding.squeeze().detach().cpu()
            else:
                try:
                    image_sam = torch.from_numpy(np.load(sam_embedding_save_path))
                except:
                    print("SAM embedding loading have bugs, please check!")
        else:
            image_sam = torch.zeros(256, 64, 64).to(self.device)


        for transform in self.mask_transforms:
            mask_sam = transform(np.expand_dims(mask_iter,-1).astype(np.float32))
            mask_sam_interpolate = F.interpolate(mask_sam[None], size = (512, 512), mode='nearest').float() # [1, 1, 512, 512]
        target['img_mask'] = mask_sam_interpolate

        # transform for other part of framework
        for transform in self.transforms:
            image, target = transform(image, target)

        # For BERT/CLIP
        examples = read_examples(target['phrase'], idx)
        try:
            features = convert_examples_to_features_clip(
                examples=examples, tokenizer=self.tokenizer, prompt_templates=['{}, a type of {}.'], model=None, cat_name=None)
        except:
                print(target['phrase'])
        word_id = features[0].input_ids
        word_mask = features[0].input_mask

        target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)
        if self.dataset == 'phrase_cut':
            target['task_id'] = task_id


        if 'mask' in target:
            mask = target.pop('mask')
            img_mask = target.pop('img_mask')
            return image, mask, target, image_sam, img_mask

        return image, target

    def load_img_feats(self, idx):
        img_path=None
        if self.dataset in ['unc','unc+','gref', 'gref_umd']:
            img_path=os.path.join(self.im_dir,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        elif self.dataset=='referit':
            img_path = os.path.join(self.im_dir, '%d.jpg' % self.refs_anno[idx]['iid'])
        elif self.dataset == 'merge':
            if self.refs_anno[idx]['data_source']=='coco':
                iid='COCO_train2014_%012d.jpg'%int(self.refs_anno[idx]['iid'].split('.')[0])
                img_path = os.path.join(self.im_dir_coco, iid)
            elif self.refs_anno[idx]['data_source']=='flickr':
                iid=self.refs_anno[idx]['iid']
                img_path = os.path.join(self.img_dir_flickr, iid)
            elif self.refs_anno[idx]['data_source']=='vg':
                iid=self.refs_anno[idx]['iid']
                img_path = os.path.join(self.img_dir_vg, iid)
            else:
                iid=self.refs_anno[idx]['iid']
                img_path = os.path.join(self.img_dir_referit, iid)
        elif self.dataset=='phrase_cut':
            iid=self.refs_anno[idx]['iid']
            img_folder = self.refs_anno[idx]['img_folder']
            img_path = os.path.join(self.img_dir_vg, img_folder, str(iid)+'.jpg')
        elif self.dataset in ['grefcoco']:
            img_path=os.path.join(self.im_dir,'COCO_train2014_%012d.jpg'%self.refs_anno[idx]['iid'])
        else:
            assert NotImplementedError

        try:
            image=Image.open(img_path).convert('RGB')
        except:
            print(img_path)
    
        if self.dataset in ['unc','unc+','gref', 'gref_umd', 'referit']:
            mask = np.load(os.path.join(self.mask_path[self.dataset],'%d.npy'%self.refs_anno[idx]['mask_id']))
        elif self.dataset in ['phrase_cut']:
            mask = np.load(os.path.join(self.mask_path[self.dataset],'%s.npy'%self.refs_anno[idx]['mask_id']))
        elif self.dataset in ['grefcoco']:
            mask = np.load(os.path.join(self.mask_path[self.dataset],'%d.npy'%self.refs_anno[idx]['mask_id']))
        else:
            mask=np.zeros([image.size[0],image.size[1]],dtype=np.uint8)

        if self.is_pretrain or self.dataset == 'phrase_cut':
            box = self.convert_bbox[idx]
        else:
            box = self.convert_bbox[self.refs_anno[idx]['mask_id']]

        if self.dataset == 'phrase_cut':
             return image,mask,box, self.task_id[idx],  img_path
        else:
            return image,mask,box, img_path

    def load_refs(self, idx):
        refs = self.refs_anno[idx]['refs']
        ref=refs[np.random.choice(len(refs))]
        return ref
