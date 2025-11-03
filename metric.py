import os
import pickle
import torch
import trimesh #加载处理3Dmesh
from .FLAME import FLAME
import numpy as np
from tqdm import tqdm



#封装Metric类（初始化、指标计算、可视化……）
class Metric:
    def __init__(self, device=None):
        #计算设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #初始化FLAME模型（从参数生成面部顶点）
        self.flame_model = FLAME()

        # 加载FLAME模型的面部区域掩码（用于区分嘴唇、眼睛等区域）
        with open('assets/FLAME_masks/FLAME_masks.pkl', 'rb') as f:
            flame_masks = pickle.load(f, encoding="latin1") #掩码数据
        self.lip_indices = flame_masks.get('lips', []) #提取嘴唇区域的顶点索引

        # 加载头部模板网格，区分上唇和下唇顶点
        mesh = trimesh.load('assets/head_template.obj')
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32) # 转换顶点为张量
        median_y = torch.median(vertices[self.lip_indices, 1]) # 计算嘴唇顶点y坐标的中位数（用于区分上下唇）

        # 基于y坐标中位数划分上唇和下唇顶点索引
        self.upper_lip_indices = [idx for idx in self.lip_indices if vertices[idx][1] > median_y]
        self.lower_lip_indices = [idx for idx in self.lip_indices if vertices[idx][1] <= median_y]

        # 定义上脸区域（眼睛、额头等）的关键区域
        upper_face_keys = [
            'eye_region', 'left_eyeball', 'right_eyeball',
            'right_eye_region', 'left_eye_region', 'forehead'
        ]
        # 情绪标签到索引的映射（用于情绪分类评估）
        self.emo2idx = {
            'angry': 0,
            'disgusted': 1,
            'contempt': 2,
            'fear': 3,
            'happy': 4,
            'sad': 5,
            'surprised': 6,
            'neutral': 7
        }

        # 收集上脸区域的所有顶点索引
        upper_face_vertices = set()
        for key in upper_face_keys:
            if key in flame_masks:
                upper_face_vertices.update(flame_masks[key]) # 合并多个区域的顶点
        self.upper_face_indices = sorted(list(upper_face_vertices))  # 转换为有序列表
    
    #计算预测网格与真实网格之间的平均顶点误差（VE）
    @staticmethod
    def vertex_error(pred_mesh: torch.Tensor, gt_mesh: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.norm(pred_mesh - gt_mesh, dim=1))

    def compute_lve(self, predicted_vertices: torch.Tensor, ground_truth_vertices: torch.Tensor) -> torch.Tensor:
        lve_per_frame = []
        # 遍历每帧
        for frame_idx in range(predicted_vertices.shape[0]):
            # 提取当前帧的嘴唇顶点（仅使用预定义的嘴唇索引）
            pred_lip = predicted_vertices[frame_idx][self.lip_indices]
            gt_lip = ground_truth_vertices[frame_idx][self.lip_indices]
             # 计算每个嘴唇顶点的L2误差
            l2_errors = torch.norm(pred_lip - gt_lip, dim=1)
            # 记录当前帧的最大嘴唇顶点误差（LVE）
            lve_per_frame.append(torch.max(l2_errors))
        return torch.stack(lve_per_frame)# 合并所有帧的结果为张量
    
    # 计算情绪分类的准确率，评估预测情绪与真实情绪的匹配程度。
    def compute_emotion_accuracy(self,pred_logits, gt_label):
        # Convert ground truth labels to indices using emo2idx
        gt_label_class = torch.tensor([self.emo2idx[label] for label in gt_label], device=pred_logits.device)
        
        # Apply softmax to the predicted logits to get probabilities
        pred_probs = torch.softmax(pred_logits, dim=-1)
        
        # Get the predicted class (highest probability) for each sample
        pred_class = torch.argmax(pred_probs, dim=-1)
        
        # Check if the predicted class matches the ground truth label (element-wise comparison)
        accuracy = torch.mean((pred_class == gt_label_class).float())
        
        return accuracy.item()  # Returns a scalar value (float)
    
    

    # 通过 t-SNE 算法将高维的情绪 logits 降维到 2D 空间，可视化不同情绪类别的聚类效果，评估模型对情绪特征的区分能力。
    def visualize_emotions_tsne(self,logits_list, labels_list, save_path='tsne_emotion_plot.png',
                                perplexity=30, learning_rate=200, figsize=(8, 6), title="t-SNE Visualization of Emotion Logits"):
        """
        Visualizes emotion logits using t-SNE and saves the plot.

        Parameters:
        - logits_list: List of emotion logits (NumPy arrays or tensors)
        - labels_list: List of ground truth emotion labels
        - save_path: Path to save the figure
        - perplexity: t-SNE perplexity parameter
        - learning_rate: t-SNE learning rate
        - figsize: Size of the matplotlib figure
        - title: Title of the plot
        """
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        # Convert to NumPy arrays
        gt_label_class = torch.tensor([self.emo2idx[label] for label in labels_list])
        logits_array = np.array(logits_list)
        labels_array = np.array(gt_label_class)

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
        tsne_results = tsne.fit_transform(logits_array)

        # Plot
        plt.figure(figsize=figsize)
        for label in np.unique(labels_array):
            indices = labels_array == label
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {label}', alpha=0.6)

        plt.title(title)
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        

    # 计算嘴巴开合偏差（MOD），评估预测的嘴唇开合程度与真实值的差异（反映说话 / 表情的嘴部动态精度）
    @staticmethod
    def compute_mod_from_mesh(pred_meshes, gt_meshes, upper_lip_indices, lower_lip_indices):
        mod_total = 0.0
        T = len(pred_meshes)
        for t in range(T):
            pred_upper = pred_meshes[t][upper_lip_indices]
            pred_lower = pred_meshes[t][lower_lip_indices]
            gt_upper = gt_meshes[t][upper_lip_indices]
            gt_lower = gt_meshes[t][lower_lip_indices]
            pred_opening = torch.norm(pred_upper - pred_lower, dim=1).mean()
            gt_opening = torch.norm(gt_upper - gt_lower, dim=1).mean()
            mod_total += torch.abs(pred_opening - gt_opening)
        return mod_total / T

    # 计算面部动态偏差（FDD），评估上脸区域（如眼睛、额头）运动的动态波动性与真实值的差异（反映表情动态的自然度）
    @staticmethod
    def compute_fdd(pred_meshes, gt_meshes, upper_face_indices):
        T = len(pred_meshes)
        V = len(upper_face_indices)

        pred_seq = torch.stack([frame[upper_face_indices] for frame in pred_meshes])
        gt_seq = torch.stack([frame[upper_face_indices] for frame in gt_meshes])

        pred_motion = torch.norm(pred_seq[1:] - pred_seq[:-1], dim=2)
        gt_motion = torch.norm(gt_seq[1:] - gt_seq[:-1], dim=2)

        pred_std = torch.std(pred_motion, dim=0)
        gt_std = torch.std(gt_motion, dim=0)

        fdd = torch.norm(pred_std - gt_std) / V
        return fdd

    def metric_main(self, predicted_folder, ground_truth_folder):
        # 将FLAME模型放到GPU并设为评估模式（关闭dropout等训练相关层）
        self.flame_model.cuda().eval()
        # 获取预测和真实参数文件列表（按文件名排序，确保对应）
        pred_files = sorted(os.listdir(predicted_folder))
        gt_files = sorted(os.listdir(ground_truth_folder))

        total_ve = 0.0
        total_lve = 0.0
        total_mod = 0.0
        total_fdd = 0.0
        total_emotion_accuracy = 0.0
        num_samples = len(pred_files)
        all_pred_logits = []
        all_gt_labels = []

        for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=num_samples):
            with open(os.path.join(predicted_folder, pred_file), 'rb') as f:
                pred_params = pickle.load(f)
            with open(os.path.join(ground_truth_folder, gt_file), 'rb') as f:
                gt_params = pickle.load(f)
                
            # Convert all values in gt_params['params_template'] to tensors
            for key, value in gt_params['params_template'].items():
                if not isinstance(value, torch.Tensor):
                    gt_params['params_template'][key] = torch.tensor(value, dtype=torch.float32).cuda()

            # Convert all values in pred_params to tensors
            for key, value in pred_params.items():
                if isinstance(value, np.ndarray):
                    pred_params[key] = torch.tensor(value, dtype=torch.float32).cuda()


            flame_target = self.flame_model.forward(gt_params['params_template'], zero_pose=True)['vertices']*1000
            params_pred = gt_params['params_template'].copy()
            params_pred['expression_params'] = pred_params['expression_params']
            params_pred['eyelid_params'] = pred_params['eyelid_params']
            params_pred['jaw_params'] = pred_params['jaw_params']
            # Get the emotion logit from the predicted params
            pred_emotion_logit = pred_params['emo_label']  # Example for 5 classes
            gt_emotion_label = gt_params['emo_label']  # 0 is the default index (e.g., happy)
            
            
            
            # 用FLAME模型生成预测顶点
            pred_vertices = self.flame_model.forward(params_pred, zero_pose=True)['vertices']*1000
            
            # 计算当前样本的各项指标
            ve = self.vertex_error(pred_vertices.view(-1, 3), flame_target.view(-1, 3))
            lve = self.compute_lve(pred_vertices, flame_target).mean()
            mod = self.compute_mod_from_mesh(pred_vertices, flame_target,
                                            self.upper_lip_indices, self.lower_lip_indices)
            fdd = self.compute_fdd(pred_vertices, flame_target, self.upper_face_indices)
            emotion_accuracy = self.compute_emotion_accuracy(pred_emotion_logit, gt_emotion_label)
            
            
            
            total_ve += ve.item()
            total_lve += lve.item()
            total_mod += mod.item()
            total_fdd += fdd.item()
            total_emotion_accuracy += emotion_accuracy
            # Collect logits and labels for t-SNE
            all_pred_logits.extend(pred_params['emo_label'].cpu().numpy())
            all_gt_labels.extend(gt_params['emo_label'])

        self.visualize_emotions_tsne(all_pred_logits,all_gt_labels)
        print(f"Vertex Error (VE): {total_ve / num_samples:.4f}")
        print(f"Lip Vertex Error (LVE): {total_lve / num_samples:.4f}")
        print(f"Mouth Opening Deviation (MOD): {total_mod / num_samples:.4f}")
        print(f"Facial Dynamics Deviation (FDD): {total_fdd / num_samples:.4f}")
        print(f"Emotion Classification Accuracy: {total_emotion_accuracy / num_samples:.4f}")



if __name__ == '__main__':
    metric = Metric()
    metric.metric_main('predicted_params','ground_truth_params')
    pass
    
