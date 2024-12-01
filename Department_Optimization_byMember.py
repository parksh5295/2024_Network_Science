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
#edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_2_edges.csv', header=None)
#edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_3_edges.csv', header=None)
#edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_4_edges.csv', header=None)
#edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_5_edges.csv', header=None)
edges_df.columns = ['source', 'target', 'weight']

edges_df['source'] = pd.to_numeric(edges_df['source'], errors='coerce')
edges_df['target'] = pd.to_numeric(edges_df['target'], errors='coerce')
edges_df = edges_df.dropna(subset=['source', 'target'])

# 'number', 'members' 컬럼을 가진 CSV 파일 로드
members_df = pd.read_csv('./employees_per_department.csv', header=None, encoding='ascii')  # 'number', 'members' 컬럼이 있다고 가정
members_df.columns = ['number', 'num_employees'] 
members_df['number'] = pd.to_numeric(members_df['number'], errors='coerce', downcast='integer')

# 부서별 member 수를 nodes_df에 추가 (number로 매칭)
nodes_df = nodes_df.merge(members_df[['number', 'num_employees']], on='number', how='left')
nodes_df['num_employees'] = pd.to_numeric(nodes_df['num_employees'], errors='coerce')

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

# 부서 성과지표와 인원 관계 설정 (예시)
a = 0.7  # 인원 수에 대한 성과 기여도
b = 0.1  # 인원 수의 제곱에 대한 성과 기여도 (비효율성)

# 부서 성과지표의 조정: 인원 수에 의한 성과
nodes_df['adjusted_performance'] = (
    a * nodes_df['num_employees'] - b * nodes_df['num_employees']**2
)

# 성과지표를 기반으로 목표 함수 설정
c = -nodes_df['adjusted_performance'].values  # 부서 성과를 최적화 목표 함수로 설정

# 제약 조건: 부서가 하나의 그룹에만 속하도록 하기
num_departments = len(nodes_df)
A_eq = []
b_eq = []

# 각 부서가 하나의 그룹에만 속하도록 하는 제약 추가
for i in range(num_departments):
    row = [1 if i == j else 0 for j in range(num_departments)]  # 그룹할 때마다 1, 다른 그룹 0
    A_eq.append(row)
    b_eq.append(1)  # 각 부서는 하나의 그룹에만 속해야 함

# 선형 계획법 문제 풀기
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

# 최적화된 그룹 배치 결과 출력
print("최적화된 그룹 배치 결과:", result.x)

# 최적화된 그룹 배치 결과 시각화
group_assignment = result.x  # 부서의 그룹 배정 결과
plt.scatter(range(num_departments), group_assignment, c=group_assignment, cmap='viridis')
plt.xlabel('Department')
plt.ylabel('Group Assignment')
plt.title('Optimized Department Grouping based on Employees and Performance')
plt.show()
