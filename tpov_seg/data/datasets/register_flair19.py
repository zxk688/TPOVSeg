import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


FLAIR19_CATEGORIES = [
    {'color': [128, 0, 0], 'isthing': 1, 'id': 0, 'name': 'building', 'trainId': 0},
    {'color': [192, 192, 128], 'isthing': 1, 'id': 1, 'name': 'pervious_surface', 'trainId': 1},
    {'color': [128, 64, 128], 'isthing': 1, 'id': 2, 'name': 'impervious_surface', 'trainId': 2},
    {'color': [160, 64, 0], 'isthing': 1, 'id': 3, 'name': 'bare_soil', 'trainId': 3},
    {'color': [0, 0, 142], 'isthing': 1, 'id': 4, 'name': 'water', 'trainId': 4},
    {'color': [0, 128, 0], 'isthing': 1, 'id': 5, 'name': 'coniferous', 'trainId': 5},
    {'color': [128, 128, 0], 'isthing': 1, 'id': 6, 'name': 'deciduous', 'trainId': 6},
    {'color': [64, 128, 32], 'isthing': 1, 'id': 7, 'name': 'brushwood', 'trainId': 7},
    {'color': [192, 0, 128], 'isthing': 1, 'id': 8, 'name': 'vineyard', 'trainId': 8},
    {'color': [64, 64, 128], 'isthing': 1, 'id': 9, 'name': 'herbaceous_vegetation', 'trainId': 9},
    {'color': [128, 0, 192], 'isthing': 1, 'id': 10, 'name': 'agricultural_land', 'trainId': 10},
    {'color': [192, 128, 64], 'isthing': 1, 'id': 11, 'name': 'plowed_land', 'trainId': 11},
    {'color': [0, 64, 64], 'isthing': 1, 'id': 12, 'name': 'swimming_pool', 'trainId': 12},
    {'color': [128, 128, 128], 'isthing': 1, 'id': 13, 'name': 'snow', 'trainId': 13},
    {'color': [64, 0, 0], 'isthing': 1, 'id': 14, 'name': 'clear_cut', 'trainId': 14},
    {'color': [0, 128, 128], 'isthing': 1, 'id': 15, 'name': 'mixed', 'trainId': 15},
    {'color': [128, 64, 0], 'isthing': 1, 'id': 16, 'name': 'ligneous', 'trainId': 16},
    {'color': [64, 128, 128], 'isthing': 1, 'id': 17, 'name': 'greenhouse', 'trainId': 17},
    {'color': [64, 64, 64], 'isthing': 1, 'id': 18, 'name': 'other', 'trainId': 18},
]

def _get_flair19_meta():
    """
    为FLAIR-19数据集生成元数据。
    """
    stuff_ids = [k["id"] for k in FLAIR19_CATEGORIES]
    stuff_dataset_id_to_contiguous_id = {k["id"]: k["trainId"] for k in FLAIR19_CATEGORIES}
    stuff_classes = [k["name"] for k in FLAIR19_CATEGORIES]
    stuff_colors = [k["color"] for k in FLAIR19_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_all_flair19(root):
    """
    注册FLAIR-19数据集。
    """
    root = './datasets/flair'  # 数据集根目录
    meta = _get_flair19_meta()
    
    # 注册训练集和验证集
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations/train"),
        ("test", "images/val", "annotations/val"), 
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        
            
        dataset_name = f"flair19_{name}_sem_seg"
        
        DatasetCatalog.register(
            dataset_name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="tif", image_ext="tif")
        )
        
        MetadataCatalog.get(dataset_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255, 
            **meta,
        )
        




_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_flair19(_root)


