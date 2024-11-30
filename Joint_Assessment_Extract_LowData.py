import pandas as pd
import numpy as np
import chardet

# CSV 파일 읽기 함수 (여러 인코딩을 시도)
def read_csv(file_path):
    # 먼저 chardet을 사용하여 인코딩 감지
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
        if encoding is None:
            encoding = 'utf-8'  # 기본값을 'utf-8'로 설정

    # 우선 chardet이 감지한 인코딩으로 시도
    try:
        df = pd.read_csv(file_path, encoding=encoding, quotechar='"', on_bad_lines='skip', header=None)
        # 열 이름을 수동으로 지정
        df.columns = ['Vision', 'Indicator_name', 'Metric_Achievement_H1', 'Metric_Achievement_H2', 'Evaluate_Nature_Metrics', 'Performance_Metric_Score_H1', 'Performance_Metric_Score_H2']
        # 열 이름에 불필요한 공백이 있으면 제거
        df.columns = df.columns.str.strip()
        return df
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError with encoding {encoding}: {e}")
        print("Trying alternative encodings...")

    # chardet 감지 인코딩에서 오류가 발생하면 다른 인코딩을 시도
    alternative_encodings = ['utf-8', 'ISO-8859-1', 'latin1', 'utf-16-le', 'windows-1252']
    
    for alt_encoding in alternative_encodings:
        try:
            df = pd.read_csv(file_path, encoding=alt_encoding, quotechar='"', on_bad_lines='skip', header=None)
            # 열 이름을 수동으로 지정
            df.columns = ['Vision', 'Indicator_name', 'Metric_Achievement_H1', 'Metric_Achievement_H2', 'Evaluate_Nature_Metrics', 'Performance_Metric_Score_H1', 'Performance_Metric_Score_H2']
            # 열 이름에 불필요한 공백이 있으면 제거
            df.columns = df.columns.str.strip()
            print(f"Successfully read with encoding {alt_encoding}")
            return df
        except UnicodeDecodeError as e:
            print(f"UnicodeDecodeError with encoding {alt_encoding}: {e}")
    
    # 그래도 오류가 나면, 마지막으로 인코딩을 확인해 보고 실패 메세지 출력
    raise Exception(f"Failed to read CSV file with available encodings.")

# Vision별로 그룹화하는 함수
def group_by_vision(df):
    return df.groupby('Vision')

# 가중치 계산 함수
def calculate_weight(row):
    metric_avg = (row['Metric_Achievement_H1'] + row['Metric_Achievement_H2']) / 2 if not np.isnan(row['Metric_Achievement_H1']) and not np.isnan(row['Metric_Achievement_H2']) else 0
    performance_avg = (row['Performance_Metric_Score_H1'] + row['Performance_Metric_Score_H2']) / 2 if not np.isnan(row['Performance_Metric_Score_H1']) and not np.isnan(row['Performance_Metric_Score_H2']) else 0
    return metric_avg * performance_avg

# Edge 생성 함수 (각 Vision에 대해)
def generate_edges_for_vision(df, start_node):
    edges = []
    for _, row in df.iterrows():
        indicators = list(map(int, row['Indicator_name'].split(','))) if isinstance(row['Indicator_name'], str) else []
        for end_node in indicators:
            if end_node != start_node:  # 시작 노드 제외
                weight = calculate_weight(row)
                edges.append([start_node, end_node, weight])
    return edges

# Vision별 Edge CSV 파일 생성 함수
def create_edge_csv_for_vision(df, vision, start_node):
    edges = generate_edges_for_vision(df, start_node)
    edge_df = pd.DataFrame(edges, columns=['Start_Node', 'End_Node', 'Weight'])
    edge_df.to_csv(f'./Department_Collaborate_Vision/vision_{vision}_edges.csv', index=False)

# Vision별로 처리하는 함수
def process_vision_data(df, start_node):
    vision_groups = group_by_vision(df)
    for vision, group_df in vision_groups:
        create_edge_csv_for_vision(group_df, vision, start_node)

# 1. 기획예산실
df1 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file1.csv')
process_vision_data(df1, 1)

# 2. 감사실
df2 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file2.csv')
process_vision_data(df2, 2)

# 3. 정책홍보실
df3 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file3.csv')
process_vision_data(df3, 3)
