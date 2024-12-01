import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random

# 각 비전별로 처리하는 반복문
for vision_num in range(1, 6):
    # 파일 읽기
    edges_df = pd.read_csv(
        f"./Department_Collaborate_Vision/vision_{vision_num}_edges.csv"
    )

    # 방향성 그래프 생성
    G = nx.DiGraph()

    # 엣지 추가
    for index, row in edges_df.iterrows():
        G.add_edge(row["Start_Node"], row["End_Node"], weight=row["Weight"])

    # 2. 각 노드의 가중치 계산 (노드의 가중치는 그 노드의 모든 degree 가중치 절반들의 합)
    node_weights = {}
    for node in G.nodes():
        # 노드의 모든 인접 엣지 가중치 절반들의 합
        total_weight = sum(G[u][v]["weight"] / 2 for u, v in G.in_edges(node)) + sum(
            G[u][v]["weight"] / 2 for u, v in G.out_edges(node)
        )
        node_weights[node] = total_weight

    # 3. 페이지랭크 계산 (기본 설정으로 계산)
    pagerank = nx.pagerank(G)

    # 4. 노드 가중치를 페이지랭크 값에 곱하기
    adjusted_node_weights = {
        node: node_weights[node] * pagerank[node] for node in G.nodes()
    }

    # 5. 네트워크의 전체 페이지랭크 값 더하기 (Total Performance of Network)
    def calculate_total_performance(G, node_weights, pagerank):
        adjusted_node_weights = {
            node: node_weights[node] * pagerank[node] for node in G.nodes()
        }
        return sum(adjusted_node_weights.values())

    # 6. 엣지 재배치 (최적화)
    def optimize_edges(G, node_weights, pagerank, max_iterations=100):
        best_performance = calculate_total_performance(G, node_weights, pagerank)
        best_edges = list(G.edges())
        nodes = list(G.nodes())

        for _ in range(max_iterations):
            # 엣지 순서를 무작위로 재배치
            random.shuffle(nodes)

            # 새로운 그래프 생성
            new_G = nx.DiGraph()

            # 수정된 부분: iterrows()에서 index와 row를 분리하여 사용
            for _, row in edges_df.iterrows():
                u = row["Start_Node"]
                v = row["End_Node"]
                weight = row["Weight"]
                new_G.add_edge(u, v, weight=weight)

            # 페이지랭크 계산
            new_pagerank = nx.pagerank(new_G)

            # Total Performance 계산
            new_performance = calculate_total_performance(
                new_G, node_weights, new_pagerank
            )

            # 성능이 향상되었으면 업데이트
            if new_performance > best_performance:
                best_performance = new_performance
                best_edges = list(new_G.edges())

        # 최적화된 엣지로 네트워크 반환
        best_G = nx.DiGraph()
        best_G.add_edges_from(best_edges)
        return best_G, best_performance

    # 엣지 재배치 최적화
    optimized_G, optimized_performance = optimize_edges(G, node_weights, pagerank)

    # 결과 출력
    print(
        f"\nVision {vision_num} - Optimized Total Performance: {optimized_performance}"
    )

    # 네트워크 시각화
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(optimized_G, k=10.0, iterations=10)
    node_sizes = [adjusted_node_weights[node] for node in optimized_G.nodes()]

    nx.draw_networkx_nodes(
        optimized_G, pos, node_size=node_sizes, node_color="lightblue"
    )
    nx.draw_networkx_edges(optimized_G, pos, alpha=0.5, width=2)
    labels = {node: int(node) for node in G.nodes()}
    nx.draw_networkx_labels(
        optimized_G, pos, labels=labels, font_size=10, font_color="black"
    )
    plt.title(f"Vision {vision_num} - Optimized Network with Adjusted Node Weights")

    # 각 비전별 그래프를 저장
    plt.savefig(f"Pagerank_vision_{vision_num}_network.png")
    plt.close()  # 메모리 관리를 위해 figure 닫기

    # 노드별 상세 정보 출력
    print(f"\nVision {vision_num} - Node Details:")
    for node in optimized_G.nodes():
        print(
            f"Node {node} - Weight: {node_weights[node]:.4f}, PageRank: {pagerank[node]:.4f}, Adjusted Weight: {adjusted_node_weights[node]:.4f}"
        )
