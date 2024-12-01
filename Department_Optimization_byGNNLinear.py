import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# 'number', 'id' 컬럼을 가진 CSV 파일 로드
nodes_df = pd.read_csv('./Department_Collaborate_node.csv', header=None)  # 'number', 'id' 컬럼이 있다고 가정
nodes_df.columns = ['number', 'id']

# 'source', 'target', 'weight' 컬럼을 가진 엣지 파일 로드
edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_1_edges.csv', header=None)  # 'source', 'target', 'weight' 컬럼이 있다고 가정
#edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_2_edges.csv')
#edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_3_edges.csv')
#edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_4_edges.csv')
#edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_5_edges.csv')
edges_df.columns = ['source', 'target', 'weight']

edges_df['source'] = pd.to_numeric(edges_df['source'], errors='coerce')
edges_df['target'] = pd.to_numeric(edges_df['target'], errors='coerce')
edges_df = edges_df.dropna(subset=['source', 'target'])

# 네트워크 그래프 생성
G = nx.Graph()

# 'nodes_df'에 있는 'id'와 'number' 정보를 노드로 추가
for _, row in nodes_df.iterrows():
    G.add_node(row['id'], number=row['number'])

# 'edges_df'에 있는 'source', 'target', 'weight' 정보를 엣지로 추가
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

# 중심성 계산 (예: Degree centrality)
centrality = nx.degree_centrality(G)

# 중심성 값을 'centrality' 컬럼으로 추가
nodes_df['centrality'] = nodes_df['id'].map(centrality)

# GNN 모델 정의
class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        # GCN (Graph Convolutional Network) 사용
        self.conv1 = pyg_nn.GCNConv(in_channels, 64)
        self.conv2 = pyg_nn.GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        # 첫 번째 그래프 합성
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        # 두 번째 그래프 합성
        x = self.conv2(x, edge_index)
        return x

# 모델을 위한 준비
nodes = nodes_df['centrality'].values  # 중심성 값을 특성으로 사용
edges = edges_df[['source', 'target']].values.T.astype(np.int64)  # 'source'와 'target'을 엣지로 변환
edges = edges.astype(np.int64)
edge_weights = edges_df['weight'].values  # 가중치 정보
edge_weights = np.array(edge_weights, dtype=np.float64)

# 데이터 텐서로 변환
x = torch.tensor(nodes, dtype=torch.float).view(-1, 1)  # 중심성 값 -> 특성 벡터로 변환
edge_index = torch.tensor(edges, dtype=torch.long)  # 엣지 인덱스
edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)  # 가중치

# PyG 데이터 객체 생성
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# GNN 모델 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 초기화
model = GNNModel(in_channels=1, out_channels=1).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 학습 루프
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    # 모델 예측
    out = model(data.x, data.edge_index)
    
    # 손실 계산 (가중치 예측 값과 실제 값 비교)
    loss = criterion(out, data.x)
    
    # 역전파
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 가중치 정보를 이용해 선형 프로그래밍 문제 정의
num_departments = len(nodes)
c = -out.squeeze().detach().cpu().numpy()  # 그래디언트 추적을 끊고 numpy()로 변환

# 제약 조건: 부서가 하나의 그룹에만 속하도록 하기
# 이 예시에서는 간단히 각 부서를 두 개의 그룹으로 나누는 문제로 가정
A_eq = []
b_eq = []

# 각 부서가 하나의 그룹에만 속하도록 하는 제약 추가
for i in range(num_departments):
    row = [1 if i == j else 0 for j in range(num_departments)]  # 그룹할 때마다 1, 다른 그룹 0
    A_eq.append(row)
    b_eq.append(1)  # 각 부서는 하나의 그룹에만 속해야 함

# 선형 프로그래밍 문제 풀기
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

print("최적화된 그룹 배치 결과:", result.x)

# 최적화된 그룹 배치 결과 시각화
group_assignment = result.x  # 부서의 그룹 배정 결과
plt.scatter(range(num_departments), group_assignment, c=group_assignment, cmap='viridis')
plt.xlabel('Department')
plt.ylabel('Group Assignment')
plt.title('Optimized Department Grouping')
plt.show()
