import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 그래프 생성 함수
def create_graph_from_csv(file_path):
    edges_df = pd.read_csv(file_path)
    G = nx.DiGraph()  # 방향성 그래프
    for _, row in edges_df.iterrows():
        G.add_edge(row['Start_Node'], row['End_Node'], weight=row['Weight'])
    return G

# 1. 두 그래프를 CSV 파일에서 생성
G1 = create_graph_from_csv('./Optimization_Clustering_edges.csv')
G2 = create_graph_from_csv('./Optimization_Pagerank_edges.csv')

# 2. 페이지랭크 계산
pagerank_G1 = nx.pagerank(G1)
pagerank_G2 = nx.pagerank(G2)

# 3. 페이지랭크 벡터 변환 (모든 노드의 집합을 기준으로)
nodes = list(set(pagerank_G1.keys()).union(set(pagerank_G2.keys())))
pagerank_vector_G1 = np.array([pagerank_G1.get(node, 0) for node in nodes]).reshape(1, -1)
pagerank_vector_G2 = np.array([pagerank_G2.get(node, 0) for node in nodes]).reshape(1, -1)

# 4. 코사인 유사도 계산
cosine_sim = cosine_similarity(pagerank_vector_G1, pagerank_vector_G2)[0][0]
print(f"Cosine Similarity between PageRank vectors: {cosine_sim:.4f}")
