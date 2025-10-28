import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

class MultiPositionGradCAM:
    """
    在CAT-Seg模型的多个位置生成分类别CAM
    """
    
    def __init__(self, model, target_positions: List[str] = None):
        """
        Args:
            model: CAT-Seg模型
            target_positions: 要分析的位置列表 ['corr_embed', 'channel_attn', 'spatial_agg']
        """
        self.model = model
        self.target_positions = target_positions or ['corr_embed', 'channel_attn', 'spatial_agg']
        
        # 存储激活值和梯度的字典
        self.activations = {}
        self.gradients = {}
        
        # 注册hook
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向hook"""
        
        # Position 1: corr_embed之后
        if 'corr_embed' in self.target_positions:
            self._register_position_hook('corr_embed', self._find_corr_embed_layer())
        
        # Position 2: channel attention之后
        if 'channel_attn' in self.target_positions:
            self._register_position_hook('channel_attn', self._find_channel_attn_layer())
        
        # Position 3: spatial aggregation之后  
        if 'spatial_agg' in self.target_positions:
            self._register_position_hook('spatial_agg', self._find_spatial_agg_layer())
    
    def _find_corr_embed_layer(self):
        """找到corr_embed后的层，这里需要根据具体模型结构调整"""
        # 由于corr_embed是函数调用，我们需要hook到conv1层
        return self.model.conv1
    
    def _find_channel_attn_layer(self):
        """找到通道注意力层"""
        # 假设是第一个aggregator layer的channel_attention
        return self.model.layers[0].channel_attention
    
    def _find_spatial_agg_layer(self):
        """找到空间聚合层"""
        # 假设是第一个aggregator layer的swin_block
        return self.model.layers[0].swin_block
    
    def _register_position_hook(self, position_name: str, target_layer):
        """为特定位置注册hook"""
        
        def forward_hook(module, input, output):
            # 存储激活值，需要根据输出格式调整
            if isinstance(output, tuple):
                self.activations[position_name] = output[0]
            else:
                self.activations[position_name] = output
        
        def backward_hook(module, grad_input, grad_output):
            # 存储梯度
            if isinstance(grad_output, tuple):
                self.gradients[position_name] = grad_output[0]
            else:
                self.gradients[position_name] = grad_output
        
        # 注册hook
        fhook = target_layer.register_forward_hook(forward_hook)
        bhook = target_layer.register_backward_hook(backward_hook)
        self.hooks.extend([fhook, bhook])
    
    def generate_multi_position_cam(
        self, 
        input_image, 
        text_features, 
        appearance_guidance,
        target_class: int,
        batch_idx: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        生成多个位置的CAM
        
        Args:
            input_image: 输入图像
            text_features: 文本特征
            appearance_guidance: 外观指导
            target_class: 目标类别索引
            batch_idx: 批次索引
            
        Returns:
            各位置的CAM字典
        """
        
        # 确保输入需要梯度
        input_image.requires_grad_(True)
        
        # 前向传播
        self.model.eval()
        output = self.model(input_image, text_features, appearance_guidance)
        
        # 计算目标分数（根据具体输出格式调整）
        if len(output.shape) == 4:  # [B, T, H, W]
            target_score = output[batch_idx, target_class].sum()
        else:
            target_score = output[batch_idx, target_class]
        
        # 反向传播
        self.model.zero_grad()
        target_score.backward()
        
        # 为每个位置计算CAM
        cams = {}
        for position in self.target_positions:
            if position in self.activations and position in self.gradients:
                cam = self._compute_single_position_cam(
                    position, target_class, batch_idx
                )
                cams[position] = cam
        
        return cams
    
    def _compute_single_position_cam(
        self, 
        position: str, 
        target_class: int, 
        batch_idx: int
    ) -> np.ndarray:
        """
        计算单个位置的CAM
        """
        
        activations = self.activations[position]
        gradients = self.gradients[position]
        
        # 根据张量维度处理
        if len(activations.shape) == 5:  # [B, C, T, H, W]
            # 提取特定类别
            act = activations[batch_idx, :, target_class, :, :]  # [C, H, W]
            grad = gradients[batch_idx, :, target_class, :, :]   # [C, H, W]
        elif len(activations.shape) == 4:  # [B, C, H, W]
            act = activations[batch_idx]
            grad = gradients[batch_idx] 
        else:
            raise ValueError(f"Unsupported activation shape: {activations.shape}")
        
        # 计算权重
        weights = torch.mean(grad, dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # 加权组合
        cam = torch.sum(weights * act, dim=0)  # [H, W]
        
        # ReLU和归一化
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.detach().cpu().numpy()
    
    def visualize_multi_position_cam(
        self, 
        original_image: np.ndarray,
        cams: Dict[str, np.ndarray],
        class_name: str = None,
        save_path: str = None
    ):
        """
        可视化多个位置的CAM结果
        """
        
        num_positions = len(cams)
        fig, axes = plt.subplots(1, num_positions + 1, figsize=(4 * (num_positions + 1), 4))
        
        # 显示原图
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 显示各位置的CAM
        position_names = {
            'corr_embed': 'Raw Correlation',
            'channel_attn': 'Channel Enhanced', 
            'spatial_agg': 'Spatially Refined'
        }
        
        for idx, (position, cam) in enumerate(cams.items()):
            # 调整CAM大小到原图尺寸
            h, w = original_image.shape[:2]
            cam_resized = self._resize_cam(cam, (h, w))
            
            # 生成热力图
            heatmap = self._apply_colormap(cam_resized)
            
            # 叠加显示
            overlay = self._overlay_cam(original_image, heatmap, alpha=0.4)
            
            axes[idx + 1].imshow(overlay)
            title = position_names.get(position, position)
            if class_name:
                title += f"\n({class_name})"
            axes[idx + 1].set_title(title)
            axes[idx + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _resize_cam(self, cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整CAM大小"""
        import cv2
        return cv2.resize(cam, target_size)
    
    def _apply_colormap(self, cam: np.ndarray) -> np.ndarray:
        """应用颜色映射"""
        import cv2
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    def _overlay_cam(self, image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """叠加CAM和原图"""
        return heatmap * alpha + image * (1 - alpha)
    
    def compare_positions(
        self,
        input_image,
        text_features, 
        appearance_guidance,
        target_class: int,
        class_name: str = None
    ) -> Dict[str, float]:
        """
        比较不同位置CAM的质量指标
        """
        
        cams = self.generate_multi_position_cam(
            input_image, text_features, appearance_guidance, target_class
        )
        
        metrics = {}
        for position, cam in cams.items():
            # 计算CAM质量指标
            metrics[position] = {
                'peak_value': float(cam.max()),
                'coverage': float((cam > 0.5).sum() / cam.size),
                'concentration': float(np.std(cam)),
                'center_mass': self._calculate_center_of_mass(cam)
            }
        
        return metrics
    
    def _calculate_center_of_mass(self, cam: np.ndarray) -> Tuple[float, float]:
        """计算CAM的质心"""
        from scipy.ndimage import center_of_mass
        return center_of_mass(cam)
    
    def cleanup(self):
        """清理hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.gradients.clear()

# 使用示例
class CATSegGradCAMAnalyzer:
    """CAT-Seg模型的完整CAM分析工具"""
    
    def __init__(self, model):
        self.model = model
        self.gradcam = MultiPositionGradCAM(model)
    
    def analyze_class_attention(
        self,
        input_image,
        text_features,
        appearance_guidance, 
        target_classes: List[int],
        class_names: List[str] = None
    ):
        """
        分析多个类别在不同位置的注意力模式
        """
        
        results = {}
        
        for i, class_idx in enumerate(target_classes):
            class_name = class_names[i] if class_names else f"Class_{class_idx}"
            
            # 生成该类别的CAM
            cams = self.gradcam.generate_multi_position_cam(
                input_image, text_features, appearance_guidance, class_idx
            )
            
            # 计算质量指标
            metrics = self.gradcam.compare_positions(
                input_image, text_features, appearance_guidance, class_idx, class_name
            )
            
            results[class_name] = {
                'cams': cams,
                'metrics': metrics
            }
        
        return results
    
    def generate_comparison_report(self, analysis_results: Dict):
        """生成对比分析报告"""
        
        print("=" * 60)
        print("CAT-Seg Multi-Position CAM Analysis Report")
        print("=" * 60)
        
        for class_name, result in analysis_results.items():
            print(f"\n📋 {class_name}:")
            print("-" * 40)
            
            metrics = result['metrics']
            for position, metric in metrics.items():
                print(f"  {position:15s}: Peak={metric['peak_value']:.3f}, "
                      f"Coverage={metric['coverage']:.3f}, "
                      f"Concentration={metric['concentration']:.3f}")
        
        # 找出最佳位置
        print(f"\n🎯 推荐分析位置:")
        print("-" * 40)
        
        position_scores = {}
        for class_name, result in analysis_results.items():
            for position, metric in result['metrics'].items():
                if position not in position_scores:
                    position_scores[position] = []
                # 综合评分（可以根据需要调整权重）
                score = metric['peak_value'] * 0.4 + metric['concentration'] * 0.6
                position_scores[position].append(score)
        
        avg_scores = {pos: np.mean(scores) for pos, scores in position_scores.items()}
        best_position = max(avg_scores.keys(), key=lambda x: avg_scores[x])
        
        print(f"  最佳位置: {best_position} (平均得分: {avg_scores[best_position]:.3f})")
        
        return best_position

# 具体使用方法
def usage_example():
    """使用示例"""
    
    # 1. 创建分析器
    analyzer = CATSegGradCAMAnalyzer(model)
    
    # 2. 分析多个类别
    target_classes = [0, 2, 4]  # 要分析的类别
    class_names = ["chair", "table", "sofa"]  # 对应的类别名
    
    results = analyzer.analyze_class_attention(
        input_image, text_features, appearance_guidance,
        target_classes, class_names
    )
    
    # 3. 生成报告
    best_position = analyzer.generate_comparison_report(results)
    
    # 4. 可视化最有趣的结果
    for class_name, result in results.items():
        analyzer.gradcam.visualize_multi_position_cam(
            original_image, result['cams'], class_name,
            save_path=f"cam_analysis_{class_name}.png"
        )
    
    # 5. 清理
    analyzer.gradcam.cleanup()

if __name__ == "__main__":
    usage_example()









"""

好的，那么我现在想要在这三个地方去做分类别的cam：
1.Aggregator的forward中，corr_embed = self.corr_embed(corr)这一行代码之后，这个是相似性计算之后得到的五维度张量，B,C,T,H,W
2.AggregatorLayer的forward中，        
x = self.channel_attention(x, text_guidance)        
x = self.swin_block(x, appearance_guidance)
这两个之后。这两个分别是通道类别聚合器和空间聚合器。
或者叫做cost slice？我不知道和cam有没有区别

"""
