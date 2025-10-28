import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

WHU_CATEGORIES = [
    {'color': [0, 0, 0], 'isthing': 0, 'id': 0, 'name': 'background', 'trainId': 0},
    {'color': [255, 255, 255], 'isthing': 1, 'id': 1, 'name': 'building', 'trainId': 1},
]

def _get_loveda_meta():
    stuff_ids = [(k["id"]) for k in WHU_CATEGORIES]
    assert len(stuff_ids) == 2, len(stuff_ids)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in WHU_CATEGORIES]
    stuff_colors = [k["color"] for k in WHU_CATEGORIES]                                    #添加颜色提取
    
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,                                                      #添加颜色的注册
    }
    return ret

def register_all_potsdam(root):
    root = './datasets/whu'
    meta = _get_loveda_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations_detectron2/train"),
        ("test", "images/val", "annotations_detectron2/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"whu_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="tif")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_potsdam(_root)
