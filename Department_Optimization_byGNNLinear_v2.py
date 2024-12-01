import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import torch.nn.init as init

# 1. CSV 파일 로드 (부서 정보 노드, 엣지 정보)
nodes_df = pd.read_csv('./Department_Collaborate_node.csv')  # 'number', 'id'
nodes_df.columns = ['number', 'id']
edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_2_edges.csv', header=None)  # 'source', 'target', 'weight'
edges_df.columns = ['source', 'target', 'weight']

# number -> node_id 매핑 테이블 생성
number_to_node_id = dict(zip(nodes_df['number'], nodes_df['id']))
# 엣지 파일의 Start_Node와 End_Node를 number에서 node_id로 변환
edges_df['source'] = edges_df['source'].map(number_to_node_id)
edges_df['target'] = edges_df['target'].map(number_to_node_id)

# 2. 노드와 엣지 데이터 정리
edges_df['source'] = pd.to_numeric(edges_df['source'], errors='coerce')
edges_df['target'] = pd.to_numeric(edges_df['target'], errors='coerce')
edges_df.dropna(subset=['source', 'target'], inplace=True)
print(edges_df.head())
print(edges_df.isna().sum())  # NaN 값이 있는지 확인

connected_nodes = set(edges_df['source']).union(set(edges_df['target']))
nodes_df_filtered = nodes_df[nodes_df['id'].isin(connected_nodes)]

# 3. 노드 인덱스를 재배열하여 사용
id_to_new_index = {node_id: idx for idx, node_id in enumerate(nodes_df_filtered['id'].unique())}
nodes_df_filtered['new_index'] = nodes_df_filtered['id'].map(id_to_new_index)

edges_df_filtered = edges_df.copy()
edges_df_filtered['source'] = edges_df_filtered['source'].map(id_to_new_index)
edges_df_filtered['target'] = edges_df_filtered['target'].map(id_to_new_index)

# 매핑된 후의 edges 출력
print("Edges after remapping to new indexes:")
print(edges_df_filtered.head())  # 확인용

# 4. 네트워크 그래프 생성 및 중심성 계산
G = nx.Graph()
for _, row in edges_df_filtered.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['weight'])

centrality = nx.degree_centrality(G)

# 중심성 계산에 맞는 인덱스 보장 (중복 제거 후 정렬된 인덱스 사용)
nodes_df_filtered = nodes_df_filtered.drop_duplicates(subset='new_index').reset_index(drop=True)
print("New Indexes in nodes_df_filtered:", nodes_df_filtered['new_index'].unique())
print("Centrality Dictionary Keys:", centrality.keys())

# 중심성 값이 없는 노드에 0을 채움
nodes_df_filtered['centrality'] = nodes_df_filtered['new_index'].map(centrality).fillna(0)

# 5. PyTorch Geometric 데이터 준비
x = torch.tensor(nodes_df_filtered['centrality'].values, dtype=torch.float).view(-1, 1)  # 중심성을 특성으로 사용
edge_index = torch.tensor(edges_df_filtered[['source', 'target']].values.T, dtype=torch.long)
edge_attr = torch.tensor(edges_df_filtered['weight'].values, dtype=torch.float).view(-1, 1)  # 가중치
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# 6. GNN 모델 정의
class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 64)
        self.conv2 = pyg_nn.GCNConv(64, out_channels)
        self._initialize_weights()

    def _initialize_weights(self):
        init.xavier_normal_(self.conv1.lin.weight)
        init.xavier_normal_(self.conv2.lin.weight)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 7. 학습 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNNModel(in_channels=1, out_channels=1).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 8. 학습 루프
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out, data.edge_attr)  # 예측값을 가중치와 비교
    if torch.isnan(loss):
        print(f"NaN detected at epoch {epoch}")
        break
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 9. 선형 프로그래밍 문제 정의 (가중치 최소화)
num_departments = len(nodes_df_filtered)
c = out.squeeze().detach().cpu().numpy().tolist()
A_eq = np.identity(num_departments)
b_eq = np.ones(num_departments)
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

# 10. 결과 시각화
group_assignment = result.x
plt.scatter(range(num_departments), group_assignment, c=group_assignment, cmap='viridis')
plt.xlabel('Department')
plt.ylabel('Group Assignment')
plt.title('Optimized Department Grouping')
plt.show()
