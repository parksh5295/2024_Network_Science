import pandas as pd
import numpy as np
import networkx as nx
from scipy.optimize import linprog


edges_df = pd.read_csv('./Department_Collaborate_Vision/vision_1_edges.csv')

# 네트워크 그래프 생성
G = nx.from_pandas_edgelist(edges_df, 'department_1', 'department_2', ['weight'])

# 각 부서의 centrality 계산 (중요도)
centrality = nx.betweenness_centrality(G, weight='weight')

# 부서별 중요도에 따라 예산 배분을 최적화
num_departments = len(centrality)
total_budget = 10000  # 예시로 총 예산 설정

# 부서 중요도를 목표 함수에 반영 (중요도가 높은 부서에 더 많은 예산 배분)
c = [-centrality[dept] for dept in centrality]  # 중요도를 기준으로 최적화

# 예산의 합이 총 예산을 넘지 않도록 제약 조건 설정
A_eq = np.ones((1, num_departments))  # 예산 합이 총 예산이 되도록
b_eq = [total_budget]

# 각 부서에 할당되는 예산의 범위 설정 (0 이상)
bounds = [(0, total_budget) for _ in range(num_departments)]

# 선형 계획법을 사용한 최적화
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# 결과 출력
if result.success:
    print("최적화된 예산 배분 결과:")
    print(result.x)
else:
    print("최적화 실패:", result.message)
