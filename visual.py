import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np


ANN_PATH_MAP = {
    "unc": "data_seg/anns/refcoco.json",
    "unc+": "data_seg/anns/refcoco+.json",
    "gref": "data_seg/anns/refcocog_google.json",
    "gref_umd": "data_seg/anns/refcocog_umd.json",
    "referit": "data_seg/anns/refclef.json",
    "phrase_cut": "data_seg/anns/phrase_cut.json",
    "grefcoco": "data_seg/anns/grefcoco.json",
}

GT_MASK_ROOT_MAP = {
    "unc": "data_seg/masks/refcoco",
    "unc+": "data_seg/masks/refcoco+",
    "gref": "data_seg/masks/refcocog_google",
    "gref_umd": "data_seg/masks/refcocog_umd",
    "referit": "data_seg/masks/refclef",
    "phrase_cut": "data_seg/masks/phrase_cut_mask",
    "grefcoco": "data_seg/masks/grefcoco",
}


def build_image_path(dataset, ann, data_root):
    if dataset in ["unc", "unc+", "gref", "gref_umd", "grefcoco"]:
        name = "COCO_train2014_%012d.jpg" % ann["iid"]
        candidates = [
            osp.join(data_root, "coco", "train2014", name),
            osp.join(data_root, "other", "coco", "train2014", name),
            osp.join(data_root, "other", "images", "mscoco", "images", "train2014", name),
        ]
        for p in candidates:
            if osp.exists(p):
                return p
        return candidates[0]
    if dataset == "referit":
        return osp.join(data_root, "refclef", "%d.jpg" % ann["iid"])
    if dataset == "phrase_cut":
        return osp.join(data_root, "VG", ann["img_folder"], "%s.jpg" % ann["iid"])
    raise ValueError("Unsupported dataset: %s" % dataset)


def load_mask(path):
    if not osp.exists(path):
        return None
    mask = np.load(path)
    if mask.ndim == 3:
        mask = mask.squeeze()
    return mask


def overlay_mask(bgr_img, mask, color, alpha=0.5, threshold=None):
    if threshold is not None:
        mask_bin = mask >= threshold
    else:
        mask_bin = mask > 0
    mask_bin = mask_bin.astype(np.uint8)
    out = bgr_img.copy()
    if mask_bin.sum() == 0:
        return out
    overlay_color = np.zeros_like(out, dtype=np.uint8)
    overlay_color[:, :] = np.array(color, dtype=np.uint8)
    idx = mask_bin > 0
    out[idx] = (out[idx] * (1.0 - alpha) + overlay_color[idx] * alpha).astype(np.uint8)
    return out


def safe_ref_text(ann):
    ref = ann.get("refs", [""])[0]
    if not isinstance(ref, str):
        ref = str(ref)
    ref = ref.replace("/", "--")
    ref = ref.replace(" ", "_")
    return ref[:120]


def main():
    parser = argparse.ArgumentParser("Overlay GT and Pred masks on images")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["unc", "unc+", "gref", "gref_umd", "referit", "phrase_cut", "grefcoco"])
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--ann_path", type=str, default="")
    parser.add_argument("--gt_mask_root", type=str, default="")
    parser.add_argument("--pred_mask_dir", type=str, default="",
                        help="Directory with predicted mask .npy files")
    parser.add_argument("--pred_name_by", type=str, default="index", choices=["index", "mask_id"])
    parser.add_argument("--out_dir", type=str, default="visualization")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--pred_threshold", type=float, default=0.5)
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--save_pred", action="store_true")
    args = parser.parse_args()

    if not args.save_gt and not args.save_pred:
        args.save_gt = True
        args.save_pred = True

    ann_path = args.ann_path if args.ann_path else ANN_PATH_MAP[args.dataset]
    gt_mask_root = args.gt_mask_root if args.gt_mask_root else GT_MASK_ROOT_MAP[args.dataset]

    with open(ann_path, "r") as f:
        anno = json.load(f)
    if args.split not in anno:
        raise KeyError("split '%s' not in %s" % (args.split, ann_path))
    samples = anno[args.split]

    if args.start_index > 0:
        samples = samples[args.start_index:]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    out_gt = osp.join(args.out_dir, args.dataset, args.split, "visual", "gt_mask_with_img")
    out_pred = osp.join(args.out_dir, args.dataset, args.split, "visual", "pred_mask_with_img")
    if args.save_gt:
        os.makedirs(out_gt, exist_ok=True)
    if args.save_pred:
        os.makedirs(out_pred, exist_ok=True)

    num_img_missing = 0
    num_gt_missing = 0
    num_pred_missing = 0

    for i, ann in enumerate(samples):
        global_idx = i + args.start_index
        img_path = build_image_path(args.dataset, ann, args.data_root)
        img = cv2.imread(img_path)
        if img is None:
            num_img_missing += 1
            continue

        ref = safe_ref_text(ann)
        base_name = "%d_%s_%s" % (global_idx, ref, osp.basename(img_path))

        if args.save_gt:
            if args.dataset == "phrase_cut":
                gt_name = "%s.npy" % ann["mask_id"]
            else:
                gt_name = "%d.npy" % int(ann["mask_id"])
            gt_mask = load_mask(osp.join(gt_mask_root, gt_name))
            if gt_mask is None:
                num_gt_missing += 1
            else:
                if gt_mask.shape[:2] != img.shape[:2]:
                    gt_mask = cv2.resize(gt_mask.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                vis_gt = overlay_mask(img, gt_mask, color=(0, 255, 0), alpha=args.alpha)
                cv2.imwrite(osp.join(out_gt, base_name), vis_gt)

        if args.save_pred:
            if not args.pred_mask_dir:
                num_pred_missing += 1
            else:
                if args.pred_name_by == "index":
                    pred_name = "%d.npy" % global_idx
                else:
                    pred_name = "%s.npy" % str(ann["mask_id"])
                pred_mask = load_mask(osp.join(args.pred_mask_dir, pred_name))
                if pred_mask is None:
                    num_pred_missing += 1
                else:
                    if pred_mask.shape[:2] != img.shape[:2]:
                        pred_mask = cv2.resize(pred_mask.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    vis_pred = overlay_mask(img, pred_mask, color=(0, 0, 255), alpha=args.alpha, threshold=args.pred_threshold)
                    cv2.imwrite(osp.join(out_pred, base_name), vis_pred)

    print("Done.")
    print("dataset=%s split=%s samples=%d" % (args.dataset, args.split, len(samples)))
    print("missing_images=%d missing_gt=%d missing_pred=%d" % (num_img_missing, num_gt_missing, num_pred_missing))


if __name__ == "__main__":
    main()
