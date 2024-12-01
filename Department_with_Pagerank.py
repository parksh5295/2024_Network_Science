import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# 1. 엣지 파일 읽기 (엣지 파일은 source, target, weight 열을 포함)
edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_2_edges.csv')  # 엣지 파일

# 방향성 그래프 생성
G = nx.DiGraph()

# 엣지 추가 (source, target, weight)
for index, row in edges_df.iterrows():
    G.add_edge(row['Start_Node'], row['End_Node'], weight=row['Weight'])

# 2. 각 노드의 가중치 계산 (노드의 가중치는 그 노드의 모든 degree 가중치 절반들의 합)
node_weights = {}
for node in G.nodes():
    # 노드의 모든 인접 엣지 가중치 절반들의 합
    total_weight = sum(G[u][v]['weight'] / 2 for u, v in G.in_edges(node)) + sum(G[u][v]['weight'] / 2 for u, v in G.out_edges(node))
    node_weights[node] = total_weight

# 3. 페이지랭크 계산 (기본 설정으로 계산)
pagerank = nx.pagerank(G)

# 4. 노드 가중치를 페이지랭크 값에 곱하기
adjusted_node_weights = {node: node_weights[node] * pagerank[node] for node in G.nodes()}

# 5. 네트워크의 전체 페이지랭크 값 더하기 (Total Performance of Network)
total_performance = sum(adjusted_node_weights.values())

# 6. 결과 출력
print(f"Total Performance of Network: {total_performance}")

# 7. 네트워크 시각화
# 노드의 크기는 페이지랭크 값에 비례하게 설정
pos = nx.spring_layout(G, k=50, seed=42, scale=2.0)
node_sizes = [adjusted_node_weights[node] * 5 for node in G.nodes()]  # 노드 크기 조정

plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
labels = {node: int(node) for node in G.nodes()}  # 노드 ID를 int 형으로 변환
nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')
plt.title("Network with Adjusted Node Weights (PageRank Adjusted)")
plt.show()

# 8. 노드별 가중치 및 페이지랭크 값 출력
for node in G.nodes():
    print(f"Node {node} - Weight: {node_weights[node]:.4f}, PageRank: {pagerank[node]:.4f}, Adjusted Weight: {adjusted_node_weights[node]:.4f}")
