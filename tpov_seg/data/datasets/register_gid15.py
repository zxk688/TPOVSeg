import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# GID-15 包含1个背景类 (0) 和15个前景类 (1-15)，共16个类别
GID15_CATEGORIES = [
    {'color': [0, 0, 0], 'isthing': 1, 'id': 0, 'name': 'background', 'trainId': 0},
    {'color': [200, 0, 0], 'isthing': 1, 'id': 1, 'name': 'industrial_land', 'trainId': 1},
    {'color': [250, 0, 150], 'isthing': 1, 'id': 2, 'name': 'urban_residential', 'trainId': 2},
    {'color': [200, 150, 150], 'isthing': 1, 'id': 3, 'name': 'rural_residential', 'trainId': 3},
    {'color': [250, 150, 150], 'isthing': 1, 'id': 4, 'name': 'traffic_land', 'trainId': 4},
    {'color': [0, 200, 0], 'isthing': 1, 'id': 5, 'name': 'paddy_field', 'trainId': 5},
    {'color': [150, 250, 0], 'isthing': 1, 'id': 6, 'name': 'irrigated_land', 'trainId': 6},
    {'color': [150, 200, 150], 'isthing': 1, 'id': 7, 'name': 'dry_cropland', 'trainId': 7},
    {'color': [200, 0, 200], 'isthing': 1, 'id': 8, 'name': 'garden_plot', 'trainId': 8},
    {'color': [150, 0, 250], 'isthing': 1, 'id': 9, 'name': 'arbor_woodland', 'trainId': 9},
    {'color': [150, 150, 250], 'isthing': 1, 'id': 10, 'name': 'shrub_land', 'trainId': 10},
    {'color': [250, 200, 0], 'isthing': 1, 'id': 11, 'name': 'natural_grassland', 'trainId': 11},
    {'color': [200, 200, 0], 'isthing': 1, 'id': 12, 'name': 'artificial_grassland', 'trainId': 12},
    {'color': [0, 0, 200], 'isthing': 1, 'id': 13, 'name': 'river', 'trainId': 13},
    {'color': [0, 150, 200], 'isthing': 1, 'id': 14, 'name': 'lake', 'trainId': 14},
    {'color': [0, 200, 250], 'isthing': 1, 'id': 15, 'name': 'pond', 'trainId': 15},
]

def _get_gid15_meta():
    """
    为GID-15数据集生成元数据。
    """
    stuff_ids = [k["id"] for k in GID15_CATEGORIES]
    # 类别总数为16 (0-15)
    assert len(stuff_ids) == 16, f"期望类别总数为16 (0-15)，但获取到 {len(stuff_ids)}"
    
    # 因为id本身就是连续的，所以这个映射是 id -> id
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in GID15_CATEGORIES]
    stuff_colors = [k["color"] for k in GID15_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret

def register_all_gid15(root):
    """
    注册GID-15数据集。
    """
    root = './datasets/gid15'  # 数据集根目录
    meta = _get_gid15_meta()
    
    # 注册训练集和验证集（测试集）
    for name, image_dirname, sem_seg_dirname in [
        ("train", "images/train", "annotations/train"),
        ("test", "images/val", "annotations/val"), 
    ]:
        image_dir = os.path.join(root, image_dirname)
        gt_dir = os.path.join(root, sem_seg_dirname)
        dataset_name = f"gid15_{name}_sem_seg"
        
        DatasetCatalog.register(
            dataset_name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="tif")
        )
        
        MetadataCatalog.get(dataset_name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            # ******** 关键修正 *********
            # 使用标准的255作为忽略值，0作为背景类需要被学习
            ignore_label=255, 
            **meta,
        )

# 注册数据集
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_gid15(_root)