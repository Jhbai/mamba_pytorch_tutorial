import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaPredictor(nn.Module):
    def __init__(self, n_features, output_dim, d_model=64, d_state=16, d_conv=4, expand=2, num_layers=2):
        """
        :param n_features: 輸入特徵維度 (例如: [pH_歷史, 酸閥門, 鹼閥門])
        :param output_dim: 輸出維度 (預測的 pH 值)
        :param d_model: 模型內部隱藏層維度 (類似 LSTM 的 hidden_size)
        :param d_state: SSM 潛在狀態 x 的維度 (通常設 16 或 64)
        :param d_conv: 局部卷積核大小 (協助平滑化高頻特徵)
        :param expand: 擴張係數 (決定內部前饋網路的寬度)
        """
        super().__init__()
        
        # 1. 將輸入特徵投影到 d_model 維度
        self.input_proj = nn.Linear(n_features, d_model)
        
        # 2. 堆疊多層 Mamba Block
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(num_layers)
        ])
        
        # 3. 輸出投影層
        self.output_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        """
        x shape: (Batch_Size, Seq_Len, n_features)
        注意：與 LSTM 不同，Mamba 不需要手動維護與傳遞 h_t, c_t。
        它的 CUDA 核心會在內部自動處理所有序列依賴，並且平行算出所有時間步的輸出。
        """
        # 投影到隱藏維度
        hidden = self.input_proj(x)  # (Batch_Size, Seq_Len, d_model)
        
        # 逐層通過 Mamba Blocks
        for layer in self.layers:
            # Mamba 原生支援 (Batch, Seq_Len, Dim)
            hidden = layer(hidden)
            
        # 預測未來數值 (這裡示範直接輸出，若需 Auto-regressive 可結合前面的歷史)
        predictions = self.output_proj(hidden) # (Batch_Size, Seq_Len, output_dim)
        
        return predictions

# --- 測試代碼 ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 假設我們有一段 3600 步的歷史資料
    batch_size = 8
    seq_len = 3600 
    n_features = 3  # [pH, 酸閥, 鹼閥]
    
    model = MambaPredictor(n_features=n_features, output_dim=1, d_model=64).to(device)
    
    # 模擬輸入張量
    X = torch.randn(batch_size, seq_len, n_features).to(device)
    
    with torch.no_grad():
        # 一次性平行推論出整段序列，這就是 Mamba 取代 LSTM 的最大優勢
        outputs = model(X)
        
    print(f"輸入形狀: {X.shape}")
    print(f"輸出形狀: {outputs.shape}") 
    # 輸出應為: torch.Size([8, 3600, 1])
