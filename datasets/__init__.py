from .dataset import VGDataset
resize_size = 336
# resize_size = 224


def _infer_resize_size_from_clip(clip_pretrained, default_size=336):
    if not clip_pretrained:
        return default_size
    ckpt = str(clip_pretrained).lower()
    if "336" in ckpt:
        return 336
    if "224" in ckpt:
        return 224
    if "vit-b-16" in ckpt:
        return 224
    return default_size


def _with_resize_size(transforms, target_size):
    updated = []
    for t in transforms:
        item = t.copy()
        t_type = item.get("type")
        if t_type == "RandomResize":
            item["sizes"] = [target_size]
        elif t_type in ["NormalizeAndPad", "NormalizeAndPadCLIP"]:
            item["size"] = target_size
        updated.append(item)
    return updated


def build_dataset(test, args):
    target_size = _infer_resize_size_from_clip(
        getattr(args, "clip_pretrained", ""),
        default_size=resize_size
    )
    train_tf = _with_resize_size(args.train_transforms, target_size) if hasattr(args, "train_transforms") else None
    test_tf = _with_resize_size(args.test_transforms, target_size) if hasattr(args, "test_transforms") else None

    if test:
        print(f"test split:{args.test_split}")
        return VGDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split=args.test_split,
                         test=True,
                         transforms=test_tf,
                         max_query_len=args.max_query_len)
    else:
        return VGDataset(data_root=args.data_root,
                          split_root=args.split_root,
                          dataset=args.dataset,
                          split='train',
                          transforms=train_tf,
                          max_query_len=args.max_query_len)


train_transforms = [
    dict(type='RandomResize', sizes=[resize_size], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=resize_size, aug_translate=True)
]

test_transforms = [
    dict(type='RandomResize', sizes=[resize_size], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=resize_size, center_place=True)
]
