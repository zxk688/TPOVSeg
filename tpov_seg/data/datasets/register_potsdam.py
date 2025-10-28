import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

POTSDAM_CATEGORIES = [
    {'color': [255, 255, 255], 'isthing': 1, 'id': 0, 'name': 'impervious-surface', 'trainId': 0},
    {'color': [0, 0, 255], 'isthing': 1, 'id': 1, 'name': 'building', 'trainId': 1},
    {'color': [0, 255, 0], 'isthing': 1, 'id': 2, 'name': 'tree', 'trainId': 2},
    {'color': [0, 255, 255], 'isthing': 1, 'id': 3, 'name': 'low-vegetation', 'trainId': 3},
    {'color': [255, 255, 0], 'isthing': 1, 'id': 4, 'name': 'car', 'trainId': 4},
    {'color': [255, 0, 0], 'isthing': 1, 'id': 5, 'name': 'clutter', 'trainId': 5},
]

def _get_loveda_meta():
    stuff_ids = [(k["id"]) for k in POTSDAM_CATEGORIES]
    assert len(stuff_ids) == 6, len(stuff_ids)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in POTSDAM_CATEGORIES]
    stuff_colors = [k["color"] for k in POTSDAM_CATEGORIES]  
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,   
    }
    return ret




def register_all_potsdam(root):
    root = './datasets/potsdam'
    meta = _get_loveda_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations_detectron2/train"),
        ("test", "images/val", "annotations_detectron2/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"potsdam_{name}_sem_seg"                                #注册的名字在此处
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
