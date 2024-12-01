import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# 부서와 구성원 수를 포함한 CSV 파일 로드
members_df = pd.read_csv('./employees_per_department.csv', header=None, encoding='ascii')
members_df.columns = ['number', 'num_employees']
members_df['number'] = pd.to_numeric(members_df['number'], errors='coerce', downcast='integer')
members_df['num_employees'] = pd.to_numeric(members_df['num_employees'], errors='coerce')
print("1")

# 각 부서의 성과를 계산
a = 0.7  # 인원 수에 대한 성과 기여도
b = 0.1  # 인원 수의 제곱에 대한 성과 기여도 (비효율성)

members_df['adjusted_performance'] = (
    a * members_df['num_employees'] - b * members_df['num_employees']**2
)
print("2")

# NaN 또는 Inf가 포함된 행을 제거
members_df = members_df.dropna(subset=['adjusted_performance'])
members_df = members_df[~members_df['adjusted_performance'].apply(np.isinf)]

members_df = members_df.reset_index(drop=True)

# 목표 함수: 각 구성원의 성과를 최적화해야 하므로, 부서별 인원 수가 아니라 개별 인원들의 성과를 다뤄야 함
# 예시로 각 부서의 구성원들을 그룹에 배치하는 방식으로 접근
num_departments = len(members_df)
c = [-x for x in members_df['adjusted_performance']]  # 각 부서의 성과를 최적화하려는 목표 함수
print("3")
print(type(c))

# NaN 또는 Inf가 여전히 c에 포함되었는지 확인
if np.any(np.isnan(c)) or np.any(np.isinf(c)):
    print("경고: 'c' 벡터에 NaN 또는 Inf 값이 포함되어 있습니다.")
else:
    print("c 벡터는 정상입니다.")

# 제약 조건: 각 부서에 있는 인원들을 여러 그룹에 재배치할 수 있도록 설정
# 예시로, 부서 내 모든 인원들은 재배치될 수 있는 조건으로 설정
A_eq = []
b_eq = []

# 제약: 각 부서의 인원 수가 정확히 배치되도록 제약 조건 추가 (각 부서의 인원들이 그룹에 배치될 수 있도록)
for i in range(num_departments):
    row = [1 if j == i else 0 for j in range(num_departments)]  # 그룹 배정
    A_eq.append(row)
    b_eq.append(members_df.loc[i, 'num_employees'])  # 각 부서의 인원 수
print("4")

# 선형 계획법 문제 풀기
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
print("5")

# 최적화된 그룹 배치 결과 출력
print("최적화된 그룹 배치 결과:", result.x)

# 최적화된 그룹 배치 결과 시각화
group_assignment = result.x  # 부서의 그룹 배정 결과
plt.scatter(range(num_departments), group_assignment, c=group_assignment, cmap='viridis')
plt.xlabel('Department')
plt.ylabel('Group Assignment')
plt.title('Optimized Grouping of Employees Based on Performance')
plt.show()

# 최적화된 그룹 배치 결과를 CSV로 저장
output_df = pd.DataFrame({
    'number': members_df['number'],  # 부서 번호
    'optimized_num_employees': group_assignment  # 최적화된 그룹 배치 결과
})

output_df.to_csv('Department_Optimization_byMember_Vision1.csv', index=False, encoding='utf-8')