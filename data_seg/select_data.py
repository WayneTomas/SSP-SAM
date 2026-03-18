import debugpy
# debugpy.connect(('10.140.60.208', 6792))

import json
import random
import os
import shutil
from PIL import Image, ImageDraw

im_dir   = "data/other/coco/train2014"
save_dir = "data/refcoco+_50_imgs"
os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "images_box"), exist_ok=True)

# 画 bbox：bbox 格式 [x, y, w, h]
def draw_bbox(img, bbox):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
    return img

if __name__ == "__main__":
    path = "data_seg/anns/refcoco+.json"
    data = json.load(open(path, 'r'))
    testB = data['testB']
    rand_select = random.sample(testB, k=50)

    testC = []
    for d in rand_select:
        src = os.path.join(im_dir, f"COCO_train2014_{d['iid']:012d}.jpg")

        # 目标文件名
        base_name = os.path.basename(src)
        dst_img  = os.path.join(save_dir, "images",      base_name)
        dst_box  = os.path.join(save_dir, "images_box",  base_name.replace(".jpg", "_box.jpg"))

        # 复制原图
        shutil.copy2(src, dst_img)

        # 画 bbox 并保存
        with Image.open(src) as im:
            im_box = draw_bbox(im, d["bbox"])
            im_box.save(dst_box)

        # 写回路径字段
        d_out = dict(d)
        d_out["img_path"]      = dst_img
        testC.append(d_out)

    # 保存 JSON
    out_path = "data_seg/anns/refcoco+_testC_50.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"testC": testC}, f, indent=4, ensure_ascii=False)

    print(f"已处理 {len(testC)} 张图像，保存到 {save_dir}")
    print(f"元数据已保存到 {out_path}")