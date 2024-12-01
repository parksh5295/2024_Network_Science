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

# NaN 값을 포함하는 행을 건너뛰는 가중치 계산 함수
def calculate_weight(row):
    # 'Evaluate_Nature_Metrics'는 사용하지 않으므로 NaN이 있어도 무시하고 넘어갑니다.
    
    # Metric_Achievement_H1과 Metric_Achievement_H2의 NaN 처리
    metric_h1 = pd.to_numeric(row['Metric_Achievement_H1'], errors='coerce')
    metric_h2 = pd.to_numeric(row['Metric_Achievement_H2'], errors='coerce')
    
    # Performance_Metric_Score_H1과 Performance_Metric_Score_H2의 NaN 처리
    performance_h1 = pd.to_numeric(row['Performance_Metric_Score_H1'], errors='coerce')
    performance_h2 = pd.to_numeric(row['Performance_Metric_Score_H2'], errors='coerce')

    # Metric_Achievement_H1과 Metric_Achievement_H2, Performance_Metric_Score_H1과 Performance_Metric_Score_H2가 모두 NaN일 경우 건너뛰기
    if pd.isna(metric_h1) and pd.isna(metric_h2) and pd.isna(performance_h1) and pd.isna(performance_h2):
        return None  # NaN이 모두 있을 경우 해당 행 건너뛰기

    # Metric_Achievement_H1과 Metric_Achievement_H2의 평균값 계산 (하나가 NaN이면 다른 값만 사용)
    if pd.isna(metric_h1) and pd.notna(metric_h2):
        metric_avg = metric_h2
    elif pd.isna(metric_h2) and pd.notna(metric_h1):
        metric_avg = metric_h1
    elif pd.notna(metric_h1) and pd.notna(metric_h2):
        metric_avg = (metric_h1 + metric_h2) / 2
    else:
        metric_avg = 0  # 두 값 모두 NaN이면 0으로 처리
    
    # Performance_Metric_Score_H1과 Performance_Metric_Score_H2의 평균값 계산 (하나가 NaN이면 다른 값만 사용)
    if pd.isna(performance_h1) and pd.notna(performance_h2):
        performance_avg = performance_h2
    elif pd.isna(performance_h2) and pd.notna(performance_h1):
        performance_avg = performance_h1
    elif pd.notna(performance_h1) and pd.notna(performance_h2):
        performance_avg = (performance_h1 + performance_h2) / 2
    else:
        performance_avg = 0  # 두 값 모두 NaN이면 0으로 처리

    # 최종 가중치 계산
    return metric_avg * performance_avg

# Edge 생성 함수 (각 Vision에 대해)
def generate_edges_for_vision(df, start_node):
    edges = []
    for _, row in df.iterrows():
        # NaN 값을 포함하는 행은 건너뛰기
        weight = calculate_weight(row)
        if weight is None:
            continue  # NaN 값이 있는 행은 건너뛰고, 가중치 계산하지 않음
        
        # 'Indicator_name'이 문자열이고, 쉼표로 구분된 숫자들이 있는지 확인
        if isinstance(row['Indicator_name'], str):
            # 쉼표로 나눈 후 각 값의 공백을 제거하고, 숫자로 변환
            indicators = []
            for item in row['Indicator_name'].split(','):
                item = item.strip()  # 공백 제거
                if item.isdigit():  # 숫자일 경우에만
                    indicators.append(int(item))  # 숫자로 변환하여 추가
        else:
            indicators = []  # 'Indicator_name'이 문자열이 아니면 빈 리스트 처리
        
        # 각 end_node에 대해 Edge 추가
        for end_node in indicators:
            if end_node != start_node:  # 시작 노드를 제외한 노드들에 대해
                edges.append([start_node, end_node, weight])
    
    return edges

# Vision별 Edge CSV 파일 생성 함수
def create_edge_csv_for_vision(df, vision, start_node):
    edges = generate_edges_for_vision(df, start_node)
    if edges:
        edge_df = pd.DataFrame(edges, columns=['Start_Node', 'End_Node', 'Weight'])
        # 기존에 파일이 있으면 덧붙이기, 없으면 새로 생성
        edge_df.to_csv(f'./Department_Collaborate_Vision/vision_{vision}_edges.csv', mode='a', header=not pd.io.common.file_exists(f'./Department_Collaborate_Vision/vision_{vision}_edges.csv'), index=False)

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

# 4. 일자리경제과
df4 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file4.csv')
process_vision_data(df4, 4)

# 5. 미래전략과
df5 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file5.csv')
process_vision_data(df5, 5)

# 6. 에너지신산업학과
df6 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file6.csv')
process_vision_data(df6, 6)

# 7. 교육지원과
df7 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file7.csv')
process_vision_data(df7, 7)

# 8. 체육진흥과
df8 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file8.csv')
process_vision_data(df8, 8)

# 9. 관광과
df9 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file9.csv')
process_vision_data(df9, 9)

# 10. 문화예술과
df10 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file10.csv')
process_vision_data(df10, 10)

# 11. 환경관리과
df11 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file11.csv')
process_vision_data(df11, 11)

# 12. 도시미화과
df12 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file12.csv')
process_vision_data(df12, 12)

# 13. 공원녹지과
df13 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file13.csv')
process_vision_data(df13, 13)

# 14. 문화예술특화기획단
df14 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file14.csv')
process_vision_data(df14, 14)


# 15. 건설과
df15 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file15.csv')
process_vision_data(df15, 15)

# 16. 안전재난과
df16 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file16.csv')
process_vision_data(df16, 16)

# 17. 도시과
df17 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file17.csv')
process_vision_data(df17, 17)

# 18. 교통행정과
df18 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file18.csv')
process_vision_data(df18, 18)

# 19. 건축행정과
df19 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file19.csv')
process_vision_data(df19, 19)

# 20. 상하수도과
df20 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file20.csv')
process_vision_data(df20, 20)

# 21. 영산포발전기획단
df21 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file21.csv')
process_vision_data(df21, 21)

# 22. 총무과
df22 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file22.csv')
process_vision_data(df22, 22)

# 23. 주민생활지원과
df23 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file23.csv')
process_vision_data(df23, 23)

# 24. 사회복지과
df24 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file24.csv')
process_vision_data(df24, 24)

# 25. 세무과
df25 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file25.csv')
process_vision_data(df25, 25)

# 26. 회계과
df26 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file26.csv')
process_vision_data(df26, 26)

# 27. 시민봉사과
df27 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file27.csv')
process_vision_data(df27, 27)

# 28. 농업정책과
df28 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file28.csv')
process_vision_data(df28, 28)

# 29. 배원예유통과
df29 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file29.csv')
process_vision_data(df29, 29)

# 30. 먹거리계획과
df30 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file30.csv')
process_vision_data(df30, 30)

# 31. 농업진흥과
df31 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file31.csv')
process_vision_data(df31, 31)

# 32. 기술지원과
df32 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file32.csv')
process_vision_data(df32, 32)

# 33. 축산과
df33 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file33.csv')
process_vision_data(df33, 33)

# 34. 보건행정과
df34 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file34.csv')
process_vision_data(df34, 34)

# 35. 감염병관리과
df35 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file35.csv')
process_vision_data(df35, 35)

# 36. 건강증진과
df36 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file36.csv')
process_vision_data(df36, 36)

# 37. 빛가람시설관리사업소
df37 = read_csv('./Low_Data/Department_AnalyzeScorecardResult/file37.csv')
process_vision_data(df37, 37)