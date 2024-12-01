import networkx as nx
import pandas as pd
import random
import matplotlib.pyplot as plt

# 1. 엣지 파일 읽기 (엣지 파일은 source, target, weight 열을 포함)
edges_df = pd.read_csv(
    "./Department_Collaborate_Vision/vision_2_edges.csv"
)  # 엣지 파일

# 방향성 그래프 생성
G = nx.DiGraph()
for index, row in edges_df.iterrows():
    G.add_edge(row["Start_Node"], row["End_Node"], weight=row["Weight"])


# 2. 노드 가중치 계산 함수 (노드의 모든 degree 가중치 절반들의 합)
def calculate_node_weights(G):
    node_weights = {}
    for node in G.nodes():
        total_weight = sum(G[u][v]["weight"] / 2 for u, v in G.in_edges(node)) + sum(
            G[u][v]["weight"] / 2 for u, v in G.out_edges(node)
        )
        node_weights[node] = total_weight
    return node_weights


# 3. Total Performance 계산 함수
def calculate_total_performance(G):
    node_weights = calculate_node_weights(G)
    pagerank = nx.pagerank(G)
    adjusted_node_weights = {
        node: node_weights[node] * pagerank[node] for node in G.nodes()
    }
    return sum(adjusted_node_weights.values()), pagerank, adjusted_node_weights


# 4. 엣지 재배치 최적화 함수
def optimize_edges(G, max_iterations=1000):
    best_performance, best_pagerank, best_weights = calculate_total_performance(G)
    best_edges = list(G.edges(data=True))

    for _ in range(max_iterations):
        new_G = G.copy()

        # 1. 엣지 추가/삭제 (노드 리스트를 명시적으로 변환)
        if random.random() < 0.5:  # 무작위로 엣지 삭제
            if len(new_G.edges()) > 0:  # 엣지가 있는 경우에만 삭제
                u, v = random.choice(list(new_G.edges()))
                new_G.remove_edge(u, v)
        else:  # 무작위로 새로운 엣지 추가
            if len(new_G.nodes()) >= 2:  # 노드가 2개 이상 있어야 샘플링 가능
                u, v = random.sample(list(new_G.nodes()), 2)  # 리스트로 변환
                if not new_G.has_edge(u, v):
                    new_G.add_edge(u, v, weight=random.uniform(0.1, 1.0))

        # 2. 새로운 그래프의 Total Performance 계산
        new_performance, new_pagerank, new_weights = calculate_total_performance(new_G)

        # 3. 성능이 향상되었을 경우 업데이트
        if new_performance > best_performance:
            best_performance = new_performance
            best_edges = list(new_G.edges(data=True))
            best_pagerank = new_pagerank
            best_weights = new_weights

    # 최적화된 그래프 생성
    best_G = nx.DiGraph()
    best_G.add_edges_from([(u, v, data) for u, v, data in best_edges])
    return best_G, best_performance, best_pagerank, best_weights


# 각 비전별 그래프 생성 및 저장
for i in range(1, 6):
    # 각 비전별 엣지 파일 읽기
    edges_df = pd.read_csv(f"./Department_Collaborate_Vision/vision_{i}_edges.csv")

    # 방향성 그래프 생성
    G = nx.DiGraph()
    for index, row in edges_df.iterrows():
        G.add_edge(row["Start_Node"], row["End_Node"], weight=row["Weight"])

    # 최적화 실행
    optimized_G, optimized_performance, optimized_pagerank, optimized_weights = (
        optimize_edges(G)
    )

    # 새로운 figure 생성
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=25.0, iterations=20)
    node_sizes = [optimized_weights[node] for node in optimized_G.nodes()]

    # 그래프 그리기
    nx.draw_networkx_nodes(
        optimized_G, pos, node_size=node_sizes, node_color="lightblue"
    )
    nx.draw_networkx_edges(optimized_G, pos, alpha=0.5, width=2)
    nx.draw_networkx_labels(optimized_G, pos, font_size=10, font_color="black")
    plt.title(f"Vision {i} Network (Performance: {optimized_performance:.4f})")

    # 그래프 저장
    plt.savefig(f"Pagerank_vision_{i}_network.png", dpi=300, bbox_inches="tight")
    plt.close()  # 메모리 관리를 위해 figure 닫기

    # 결과 출력
    print(f"\n=== Vision {i} Results ===")
    print(f"Optimized Total Performance: {optimized_performance:.4f}")

    # 최적화된 엣지를 CSV 파일로 저장
    output_file = (
        f"./Optimization_Pagerank_edges_vision_{i}.csv"  # 비전별 CSV 저장 경로
    )
    optimized_edges_df = pd.DataFrame(
        [(u, v, data["weight"]) for u, v, data in optimized_G.edges(data=True)],
        columns=["Start_Node", "End_Node", "Weight"],
    )
    # CSV 파일로 저장
    optimized_edges_df.to_csv(output_file, index=False)
    print(f"Optimized edges have been saved to: {output_file}")
