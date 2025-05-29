import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import warnings
from anndata import ImplicitModificationWarning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gc
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score

# GPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
torch.backends.cudnn.benchmark = True  # 优化CUDA性能

# 忽略警告
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
warnings.filterwarnings("ignore")

# 基于PyTorch的数据预处理工具类
class TorchDataProcessor:
    def __init__(self, device):
        self.device = device
    
    def nanmean_torch(self, tensor, dim=None, keepdim=False):
        """PyTorch版本的nanmean"""
        mask = ~torch.isnan(tensor)
        if dim is None:
            return torch.sum(tensor * mask) / torch.sum(mask)
        else:
            return torch.sum(tensor * mask, dim=dim, keepdim=keepdim) / torch.sum(mask, dim=dim, keepdim=keepdim)
    
    def nanvar_torch(self, tensor, dim=None, keepdim=False):
        """PyTorch版本的nanvar"""
        mask = ~torch.isnan(tensor)
        if dim is None:
            mean_val = self.nanmean_torch(tensor)
            var_val = torch.sum(((tensor - mean_val) ** 2) * mask) / (torch.sum(mask) - 1)
            return var_val
        else:
            mean_val = self.nanmean_torch(tensor, dim=dim, keepdim=True)
            var_val = torch.sum(((tensor - mean_val) ** 2) * mask, dim=dim, keepdim=keepdim) / (torch.sum(mask, dim=dim, keepdim=keepdim) - 1)
            return var_val
    
    def fill_missing_values(self, X_tensor, batch_tensor=None):
        """使用PyTorch进行缺失值填补"""
        print("执行基于PyTorch的缺失值填补...")
        X_filled = X_tensor.clone()
        
        if batch_tensor is not None:
            unique_batches = torch.unique(batch_tensor)
            for batch in unique_batches:
                batch_mask = batch_tensor == batch
                batch_data = X_filled[batch_mask]
                
                # 计算每个特征的均值（忽略NaN）
                batch_mean = self.nanmean_torch(batch_data, dim=0)
                
                # 填补缺失值
                nan_mask = torch.isnan(batch_data)
                batch_data = torch.where(nan_mask, batch_mean.unsqueeze(0), batch_data)
                X_filled[batch_mask] = batch_data
        else:
            # 全局均值填补
            col_mean = self.nanmean_torch(X_filled, dim=0)
            X_filled = torch.where(torch.isnan(X_filled), col_mean.unsqueeze(0), X_filled)
        
        # 最终清理剩余的NaN
        X_filled = torch.where(torch.isnan(X_filled), torch.zeros_like(X_filled), X_filled)
        torch.cuda.empty_cache()  # 清理GPU内存
        return X_filled
    
    def standardize(self, X_tensor):
        """PyTorch标准化"""
        mean = torch.mean(X_tensor, dim=0, keepdim=True)
        std = torch.std(X_tensor, dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)  # 避免除零
        return (X_tensor - mean) / std, mean, std
    
    def pca_torch(self, X_tensor, n_components=200):
        """基于PyTorch的PCA实现"""
        print(f"执行PyTorch PCA降维到 {n_components} 维...")
        
        # 中心化数据
        X_centered = X_tensor - torch.mean(X_tensor, dim=0, keepdim=True)
        
        # SVD分解
        U, S, V = torch.svd(X_centered)
        
        # 取前n_components个主成分
        n_components = min(n_components, min(X_tensor.shape) - 1)
        X_pca = U[:, :n_components] * S[:n_components].unsqueeze(0)
        
        print(f"PCA完成，输出维度: {X_pca.shape}")
        return X_pca, V[:, :n_components], S[:n_components]

# 变分自编码器
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=64, dropout_rate=0.2):
        super(VariationalAutoEncoder, self).__init__()
        
        # 编码器
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 潜在空间参数
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE损失函数"""
    # 重构损失
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# 深度聚类网络
class DeepClusteringNetwork(nn.Module):
    def __init__(self, input_dim, n_clusters, hidden_dims=[512, 256, 128], dropout_rate=0.2):
        super(DeepClusteringNetwork, self).__init__()
        
        self.n_clusters = n_clusters
        
        # 特征提取器
        feature_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # 聚类中心
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, prev_dim))
        nn.init.xavier_normal_(self.cluster_centers.data)
        
        # 温度参数
        self.alpha = 1.0
        
    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        
        # 计算学生t分布（软分配）
        q = self._soft_assignment(features)
        
        return features, q
    
    def _soft_assignment(self, features):
        """计算软分配概率"""
        # 计算特征到聚类中心的距离
        distances = torch.sum((features.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        
        # 学生t分布
        q = 1.0 / (1.0 + distances / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        return q

def target_distribution(q):
    """计算目标分布P"""
    weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
    p = weight / torch.sum(weight, dim=1, keepdim=True)
    return p

# PyTorch KMeans实现
class TorchKMeans:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4, device='cuda'):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.device = device
        
    def fit_predict(self, X):
        """KMeans聚类"""
        n_samples, n_features = X.shape
        
        # 随机初始化聚类中心
        centroids = X[torch.randperm(n_samples)[:self.n_clusters]].clone()
        
        for i in range(self.max_iters):
            # 计算距离并分配
            distances = torch.cdist(X, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # 更新聚类中心
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if torch.sum(mask) > 0:
                    new_centroids[k] = torch.mean(X[mask], dim=0)
                else:
                    new_centroids[k] = centroids[k]
            
            # 检查收敛
            if torch.norm(new_centroids - centroids) < self.tol:
                break
                
            centroids = new_centroids
        
        # 最终分配
        distances = torch.cdist(X, centroids)
        labels = torch.argmin(distances, dim=1)
        
        return labels, centroids

# 基于PyTorch的UMAP近似实现
class TorchUMAP(nn.Module):
    def __init__(self, input_dim, output_dim=10, n_neighbors=15):
        super(TorchUMAP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neighbors = n_neighbors
        
        # 使用神经网络近似UMAP变换
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)
    
    def fit_transform(self, X, epochs=200, lr=1e-3):
        """训练UMAP近似模型"""
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # 构建k近邻图
        distances = torch.cdist(X, X)
        _, knn_indices = torch.topk(distances, k=self.n_neighbors + 1, largest=False)
        knn_indices = knn_indices[:, 1:]  # 排除自己
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播
            embeddings = self.forward(X)
            
            # UMAP损失（简化版）
            loss = self._umap_loss(X, embeddings, knn_indices)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f'UMAP Epoch {epoch}, Loss: {loss.item():.6f}')
        
        self.eval()
        with torch.no_grad():
            return self.forward(X)
    
    def _umap_loss(self, X, embeddings, knn_indices):
        """简化的UMAP损失函数"""
        batch_size = X.shape[0]
        
        # 高维空间距离
        high_dim_dists = torch.norm(X.unsqueeze(1) - X[knn_indices], dim=2)
        
        # 低维空间距离
        low_dim_dists = torch.norm(embeddings.unsqueeze(1) - embeddings[knn_indices], dim=2)
        
        # 吸引力损失（近邻应该在低维空间中保持接近）
        attract_loss = torch.mean(low_dim_dists)
        
        # 排斥力损失（随机采样的远距离点应该在低维空间中分离）
        n_negative = min(5, batch_size - self.n_neighbors - 1)
        if n_negative > 0:
            neg_indices = torch.randint(0, batch_size, (batch_size, n_negative), device=X.device)
            neg_low_dists = torch.norm(embeddings.unsqueeze(1) - embeddings[neg_indices], dim=2)
            repel_loss = torch.mean(1.0 / (1.0 + neg_low_dists ** 2))
        else:
            repel_loss = 0
        
        return attract_loss + repel_loss

# 数据加载函数
def load_data_to_torch():
    """加载数据并转换为PyTorch张量"""
    print("加载数据到PyTorch...")
    ad = sc.read_h5ad('final_dataset.h5ad/final_dataset.h5ad')
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(ad.X.astype(np.float32)).to(device)
    
    # 处理batch信息
    batch_tensor = None
    if 'batch' in ad.obs.columns:
        # 将batch标签转换为数值
        batch_labels = pd.Categorical(ad.obs['batch']).codes
        batch_tensor = torch.LongTensor(batch_labels).to(device)
    
    print(f"数据形状: {X_tensor.shape}")
    print(f"缺失值数量: {torch.sum(torch.isnan(X_tensor)).item()}")
    torch.cuda.empty_cache()  # 清理GPU内存
    return X_tensor, batch_tensor, ad.obs_names

# 主要训练函数
def train_vae_torch(X_tensor, epochs=100, batch_size=256, learning_rate=1e-3):
    """训练VAE"""
    print("训练变分自编码器...")
    
    input_dim = X_tensor.shape[1]
    vae = VariationalAutoEncoder(input_dim, hidden_dims=[512, 256], latent_dim=64).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    
    # 数据加载器
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for batch_data, in dataloader:
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = vae(batch_data)
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, batch_data, mu, logvar)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
        
        scheduler.step()
        
        if epoch % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            avg_recon = recon_loss_total / len(dataloader)
            avg_kl = kl_loss_total / len(dataloader)
            print(f'VAE Epoch {epoch}: Total={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}')
    
    # 获取潜在表示
    vae.eval()
    latent_representations = []
    with torch.no_grad():
        for batch_data, in dataloader:
            mu, _ = vae.encode(batch_data)
            latent_representations.append(mu)
    
    latent_data = torch.cat(latent_representations, dim=0)
    print(f"VAE训练完成，潜在空间维度: {latent_data.shape}")
    
    return latent_data, vae

def train_deep_clustering(X_tensor, n_clusters=15, epochs=200, batch_size=256, learning_rate=1e-3):
    """训练深度聚类网络"""
    print(f"训练深度聚类网络，聚类数: {n_clusters}")
    
    input_dim = X_tensor.shape[1]
    model = DeepClusteringNetwork(input_dim, n_clusters, hidden_dims=[256, 128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    
    # 使用KMeans初始化聚类中心
    print("使用KMeans初始化聚类中心...")
    kmeans = TorchKMeans(n_clusters=n_clusters, device=device)
    initial_labels, initial_centers = kmeans.fit_predict(X_tensor)
    
    # 数据加载器
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_data, in dataloader:
            optimizer.zero_grad()
            
            features, q = model(batch_data)
            
            # 计算目标分布
            p = target_distribution(q.detach())
            
            # KL散度损失
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            
            # 特征正则化
            feature_reg = torch.mean(torch.norm(features, dim=1))
            
            total_loss_batch = kl_loss + 0.01 * feature_reg
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        scheduler.step()
        
        if epoch % 25 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Deep Clustering Epoch {epoch}: Loss={avg_loss:.6f}')
    
    # 获取最终聚类结果
    model.eval()
    all_features = []
    all_q = []
    
    with torch.no_grad():
        for batch_data, in dataloader:
            features, q = model(batch_data)
            all_features.append(features)
            all_q.append(q)
    
    final_features = torch.cat(all_features, dim=0)
    final_q = torch.cat(all_q, dim=0)
    final_clusters = torch.argmax(final_q, dim=1)
    
    print("深度聚类训练完成")
    return final_clusters.cpu().numpy(), final_features, model

def ensemble_clustering_torch(X_tensor, n_clusters_list=[10, 12, 15, 18, 20]):
    """基于PyTorch的集成聚类"""
    print("执行集成聚类...")
    
    all_results = []
    
    # 方法1: 多个K值的KMeans
    for n_clusters in n_clusters_list:
        print(f"KMeans with k={n_clusters}")
        kmeans = TorchKMeans(n_clusters=n_clusters, device=device)
        labels, _ = kmeans.fit_predict(X_tensor)
        all_results.append(labels.cpu().numpy())
    
    # 方法2: UMAP + KMeans
    print("UMAP + KMeans集成...")
    umap_model = TorchUMAP(X_tensor.shape[1], output_dim=15).to(device)
    X_umap = umap_model.fit_transform(X_tensor, epochs=100)
    
    for n_clusters in n_clusters_list:
        kmeans = TorchKMeans(n_clusters=n_clusters, device=device)
        labels, _ = kmeans.fit_predict(X_umap)
        all_results.append(labels.cpu().numpy())
    
    # 集成投票
    print("计算集成结果...")
    n_samples = X_tensor.shape[0]
    ensemble_result = torch.zeros(n_samples, dtype=torch.long, device=device)
    
    # 转换为torch张量进行投票
    all_results_tensor = torch.stack([torch.tensor(result, device=device) for result in all_results])
    
    # 对每个样本进行投票
    for i in range(n_samples):
        votes = all_results_tensor[:, i]
        # 找到最频繁的标签
        unique_votes, counts = torch.unique(votes, return_counts=True)
        most_frequent_idx = torch.argmax(counts)
        ensemble_result[i] = unique_votes[most_frequent_idx]
    
    print("集成聚类完成")
    return ensemble_result.cpu().numpy()

# 主函数
def main():
    print("开始基于PyTorch的单细胞DNA甲基化聚类分析...")
    
    # 1. 数据加载
    X_tensor, batch_tensor, obs_names = load_data_to_torch()
    
    # 2. 数据预处理
    processor = TorchDataProcessor(device)
    
    # 缺失值填补
    X_filled = processor.fill_missing_values(X_tensor, batch_tensor)
    
    # 特征选择（基于方差）
    print("执行特征选择...")
    feature_vars = processor.nanvar_torch(X_filled, dim=0)
    n_top_features = min(5000, X_filled.shape[1])
    _, top_indices = torch.topk(feature_vars, k=n_top_features)
    X_selected = X_filled[:, top_indices]
    torch.cuda.empty_cache()
    
    
    print(f"特征选择完成，保留 {n_top_features} 个特征")
    
    # 标准化
    X_scaled, _, _ = processor.standardize(X_selected)
    torch.cuda.empty_cache()

    # PCA降维
    X_pca, _, _ = processor.pca_torch(X_scaled, n_components=5000)
    torch.cuda.empty_cache()

    # 3. VAE训练
    X_vae, vae_model = train_vae_torch(X_pca, epochs=2000, batch_size=256)
    torch.cuda.empty_cache()

    # 4. 深度聚类
    deep_clusters, deep_features, deep_model = train_deep_clustering(X_vae, n_clusters=15, epochs=2000)
    torch.cuda.empty_cache()

    # 5. 集成聚类
    ensemble_clusters = ensemble_clustering_torch(X_vae, n_clusters_list=[10, 12, 15, 18, 20])
    torch.cuda.empty_cache()
    
    # 6. 选择最终结果（可以根据需要选择）
    final_clusters = deep_clusters  # 或者使用 ensemble_clusters
    
    # 7. 保存结果
    print("保存聚类结果...")
    submission_df = pd.DataFrame({
        'ID': range(len(final_clusters)), 
        'TARGET': final_clusters
    })
    submission_df.to_csv("submission.csv", index=False)
    
    print(f"聚类完成!")
    print(f"识别出的聚类数: {len(torch.unique(torch.tensor(final_clusters)))}")
    print(f"各聚类样本数: {torch.bincount(torch.tensor(final_clusters)).tolist()}")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    
    print("任务完成!")

if __name__ == "__main__":
    main()

# [0, 1804, 0, 0, 0, 0, 0, 0, 0, 1524, 0, 1706, 18]
# [0, 0, 0, 0, 21, 1633, 0, 0, 0, 0, 0, 0, 1855, 1543]