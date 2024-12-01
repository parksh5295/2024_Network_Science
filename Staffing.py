import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# 부서 직렬 데이터
job_titles = ['행정직', '세무직', '전산직', '사회복지직', '공업직', '농업직', '녹지직', 
              '보건직', '의료기술직', '간호직', '환경직', '시설직', '운전직', '관리운영직', 
              '연구직', '지도직', '기타직']

departments = ['기획예산실', '감사실', '정책홍보실', '일자리경제과', '미래전략과', '에너지신산업과',
               '교육지원과', '체육진흥과', '관광과', '문화예술과', '환경관리과', '도시미화과', 
               '공원녹지과', '문화예술특화기획단', '건설과', '안전재난과', '도시과', '교통행정과',
               '건축허가과', '상하수도과', '영산포발전기획단', '총무과', '주민생활지원과', '사회복지과', 
               '세무과', '회계과', '시민봉사과', '농업정책과', '배원예유통과', '먹거리계획과', 
               '농업진흥과', '기술지원과', '축산과', '보건행정과', '감염병관리과', '건강증진과',
               '빛가람시설관리사업소']

# 부서 인원 분포 (각 직렬 별로 각 부서에 배치할 인원 수를 정의)
department_job_mapping = {
    '기획예산실': [0.5, 0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.1],
    '감사실': [0.3, 0.3, 0.1, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    '정책홍보실': [0.4, 0.2, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0.1],
    '일자리경제과': [0.6, 0, 0, 0.2, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1],
    '미래전략과': [0.5, 0, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.1],
    '에너지신산업과': [0.4, 0, 0.3, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.2],
    '교육지원과': [0.5, 0.1, 0.1, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    '체육진흥과': [0.7, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.1],
    '관광과': [0.6, 0, 0, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    '문화예술과': [0.5, 0.1, 0, 0.3, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    '환경관리과': [0.4, 0, 0.1, 0, 0.1, 0.2, 0.1, 0.1, 0, 0, 0, 0.1],
    '도시미화과': [0.4, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.1],
    '공원녹지과': [0.3, 0, 0, 0.1, 0.1, 0.4, 0.1, 0, 0, 0, 0, 0.1],
    '문화예술특화기획단': [0.6, 0, 0, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    '건설과': [0.5, 0.1, 0.2, 0, 0.1, 0.1, 0, 0, 0, 0, 0, 0.1],
    '안전재난과': [0.4, 0.1, 0.1, 0.2, 0.1, 0, 0, 0.1, 0, 0, 0, 0.1],
    '도시과': [0.6, 0.1, 0, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0],
    '교통행정과': [0.5, 0.1, 0.1, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    '건축허가과': [0.4, 0.1, 0.2, 0.1, 0, 0.1, 0, 0, 0, 0, 0, 0.1],
    '상하수도과': [0.3, 0, 0, 0.1, 0.1, 0.2, 0.1, 0, 0, 0, 0, 0.1],
    '영산포발전기획단': [0.4, 0.2, 0.1, 0, 0.1, 0, 0.1, 0, 0, 0, 0, 0.1],
    '총무과': [0.5, 0.1, 0.1, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    '주민생활지원과': [0.4, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.1],
    '사회복지과': [0.3, 0.1, 0, 0.4, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    '세무과': [0.7, 0.2, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0],
    '회계과': [0.6, 0.2, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0],
    '시민봉사과': [0.5, 0.2, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.1],
    '농업정책과': [0.3, 0, 0, 0.1, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0.1],
    '배원예유통과': [0.3, 0.2, 0, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
    '먹거리계획과': [0.5, 0, 0, 0.2, 0.1, 0, 0, 0.1, 0, 0, 0, 0.1],
    '농업진흥과': [0.4, 0, 0, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0.1],
    '기술지원과': [0.5, 0, 0.1, 0.2, 0, 0.2, 0, 0, 0, 0, 0, 0.1],
    '축산과': [0.4, 0, 0, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0.1],
    '보건행정과': [0.5, 0.1, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.1],
    '감염병관리과': [0.5, 0, 0, 0, 0, 0, 0.1, 0.2, 0, 0, 0, 0.1],
    '건강증진과': [0.4, 0, 0, 0.3, 0, 0, 0.1, 0, 0, 0, 0, 0.1],
    '빛가람시설관리사업소': [0.3, 0, 0.1, 0, 0, 0, 0.1, 0.2, 0, 0, 0, 0.1]
}

# 각 직렬에 대한 배치 비율을 기반으로 성과 기여도를 설정 (가상의 값)
performance_contributions = {
    '행정직': 0.8, '세무직': 1.0, '전산직': 0.9, '사회복지직': 0.7, '공업직': 0.6, '농업직': 0.5,
    '녹지직': 0.4, '보건직': 0.9, '의료기술직': 0.85, '간호직': 0.75, '환경직': 0.7, '시설직': 0.6,
    '운전직': 0.3, '관리운영직': 0.5, '연구직': 0.95, '지도직': 0.8, '기타직': 0.6
}

# 각 직렬별 배치할 공무원 수 (가상의 데이터)
job_counts = [522, 29, 11, 131, 43, 58, 22, 47, 23, 32, 22, 144, 35, 9, 7, 33, 37]

# 선형 계획법 문제 정의
num_departments = len(departments)
num_jobs = len(job_titles)

for department, job_mapping in department_job_mapping.items():
    if len(job_mapping) < len(job_titles):
        department_job_mapping[department] = job_mapping + [0] * (len(job_titles) - len(job_mapping))

# 목적 함수: 부서별 성과 기여도를 최대화하기 위해 각 직렬의 기여도를 배치 비율과 곱하여 최대화
c = np.zeros(num_jobs * num_departments)

for i, department in enumerate(departments):
    for j, job in enumerate(job_titles):
        c[i * num_jobs + j] = performance_contributions[job] * department_job_mapping[department][j]

# 제약 조건: 각 부서에 최소 5명 이상 배치해야 한다.
A_eq = np.zeros((num_departments, num_jobs * num_departments))
b_eq = np.ones(num_departments) * 5

for i in range(num_departments):
    for j in range(num_jobs):
        A_eq[i, i * num_jobs + j] = 1

# 부서별 공무원 배치 수에 맞게 제한을 둡니다.
A_ub = np.zeros((num_jobs, num_jobs * num_departments))
b_ub = np.array(job_counts)

for j in range(num_jobs):
    for i in range(num_departments):
        A_ub[j, i * num_jobs + j] = 1

# 부서별 공무원 수를 최적화할 수 있도록 설정합니다.
bounds = [(0, None) for _ in range(num_jobs * num_departments)]

# 최적화 문제 해결
result = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# 결과 확인
if result.success:
    print("최적화 성공")
    optimized_allocation = result.x.reshape((num_departments, num_jobs))
    print(optimized_allocation)
else:
    print("최적화 실패")

# 결과 시각화 (배치된 공무원 수 시각화)
optimized_allocation = result.x.reshape((num_departments, num_jobs))
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(optimized_allocation, cmap='viridis', aspect='auto')
ax.set_xticks(np.arange(len(job_titles)))
ax.set_yticks(np.arange(len(departments)))
ax.set_xticklabels(job_titles)
ax.set_yticklabels(departments)
plt.xticks(rotation=90)
plt.xlabel('Job Titles')
plt.ylabel('Departments')
plt.title('Optimized Allocation of Public Employees')
fig.colorbar(im)
plt.show()
