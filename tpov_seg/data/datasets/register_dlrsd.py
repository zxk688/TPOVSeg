import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

DLSRD_CATEGORIES = [
    {'color': [166, 202, 240], 'isthing': 1, 'id': 0, 'name': 'airplane', 'trainId': 0},
    {'color': [128, 128, 0], 'isthing': 1, 'id': 1, 'name': 'bare soil', 'trainId': 1},
    {'color': [0, 0, 128], 'isthing': 1, 'id': 2, 'name': 'buildings', 'trainId': 2},
    {'color': [255, 0, 0], 'isthing': 1, 'id': 3, 'name': 'cars', 'trainId': 3},
    {'color': [0, 128, 0], 'isthing': 1, 'id': 4, 'name': 'chaparral', 'trainId': 4},
    {'color': [128, 0, 0], 'isthing': 1, 'id': 5, 'name': 'court', 'trainId': 5},
    {'color': [255, 233, 233], 'isthing': 1, 'id': 6, 'name': 'dock', 'trainId': 6},
    {'color': [160, 160, 164], 'isthing': 1, 'id': 7, 'name': 'field', 'trainId': 7},
    {'color': [0, 128, 128], 'isthing': 1, 'id': 8, 'name': 'grass', 'trainId': 8},
    {'color': [90, 87, 255], 'isthing': 1, 'id': 9, 'name': 'mobile', 'trainId': 9},
    {'color': [255, 255, 0], 'isthing': 1, 'id': 10, 'name': 'pavement', 'trainId': 10},
    {'color': [255, 192, 0], 'isthing': 1, 'id': 11, 'name': 'sand', 'trainId': 11},
    {'color': [0, 0, 255], 'isthing': 1, 'id': 12, 'name': 'sea', 'trainId': 12},
    {'color': [255, 0, 192], 'isthing': 1, 'id': 13, 'name': 'ship', 'trainId': 13},
    {'color': [128, 0, 128], 'isthing': 1, 'id': 14, 'name': 'tanks', 'trainId': 14},
    {'color': [0, 255, 0], 'isthing': 1, 'id': 15, 'name': 'trees', 'trainId': 15},
    {'color': [0, 255, 255], 'isthing': 1, 'id': 16, 'name': 'water', 'trainId': 16}
]

def _get_loveda_meta():
    stuff_ids = [(k["id"]) for k in DLSRD_CATEGORIES]
    assert len(stuff_ids) == 17, len(stuff_ids)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in DLSRD_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret

def register_all_dlrsd(root):
    root = './datasets/DLRSD'
    meta = _get_loveda_meta()
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations_detectron2/train"),
        ("test", "images/val", "annotations_detectron2/val"),
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        name = f"dlrsd_{name}_sem_seg"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="tif")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_dlrsd(_root)
