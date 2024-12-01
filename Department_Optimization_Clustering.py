import networkx as nx
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from networkx.algorithms.centrality import betweenness_centrality
import community

# 현재 파일의 경로 (스크립트의 위치 기준)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 'python-louvain' 폴더를 sys.path에 추가 (현재 폴더에서 python-louvain 폴더를 참조)
louvain_path = os.path.join(current_dir, 'python-louvain')
sys.path.append(louvain_path)

# 1. CSV 파일에서 엣지 데이터 읽기
edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_5_edges.csv')  # 'edges.csv'는 source, target, weight 열이 있는 파일

# 그래프 생성 (방향성 그래프)
G = nx.DiGraph()

# 엣지 추가
for index, row in edges_df.iterrows():
    G.add_edge(row['Start_Node'], row['End_Node'], weight=row['Weight'])

# 2. Louvain 알고리즘을 통한 커뮤니티 탐지
partition = community.best_partition(G.to_undirected())  # 커뮤니티 찾기

# 3. 커뮤니티 내 엣지 가중치 강화
for u, v, data in G.edges(data=True):
    community_u = partition[u]  # 노드 u의 커뮤니티
    community_v = partition[v]  # 노드 v의 커뮤니티
    if community_u == community_v:  # 같은 커뮤니티 내 연결일 경우
        data['weight'] *= 2  # 가중치 강화 (예: 가중치를 두 배로 증가)

# 4. 중요한 노드들 간의 연결 강화 (Betweenness Centrality 기준)
centrality = betweenness_centrality(G)  # Betweenness Centrality 계산

for u, v, data in G.edges(data=True):
    if partition[u] == partition[v]:  # 같은 커뮤니티 내에서
        if centrality[u] > 0.5 and centrality[v] > 0.5:  # 중요한 노드들 간의 연결
            data['weight'] *= 2  # 연결 강화

# 5. 강화된 엣지를 CSV 파일로 저장
output_file = os.path.join(current_dir, './Optimization_Clustering_edges.csv')  # 저장할 파일 경로
enhanced_edges_df = pd.DataFrame(
    [(u, v, data['weight']) for u, v, data in G.edges(data=True)],
    columns=['Start_Node', 'End_Node', 'Weight']
)
# CSV 파일로 저장
enhanced_edges_df.to_csv(output_file, index=False)

# 5. 강화된 네트워크 시각화
# spring_layout에서 노드 간 간격을 조정하여 겹치지 않도록 설정
pos = nx.spring_layout(G, k=15.0, seed=42)  # k값을 1.0으로 설정하여 간격을 조정

# 커뮤니티별 색상 지정
colors = [partition[node] for node in G.nodes()]

# 그래프 그리기
plt.figure(figsize=(12, 12))

# 노드 색상, 크기, 위치 설정
nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.jet, node_size=500)

# 엣지 설정 (alpha=0.5로 투명도 조정)
nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)

# 레이블을 int 형으로 표시
labels = {node: int(node) for node in G.nodes()}  # 노드 ID를 int 형으로 변환
nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color='white')

# 그래프 제목
plt.title("Network with Enhanced Community Structure")
plt.show()

# 6. 최종 결과 출력 (가중치 강화된 엣지 정보)
for u, v, data in G.edges(data=True):
    print(f"Edge ({u}, {v}) - Weight: {data['weight']}")
