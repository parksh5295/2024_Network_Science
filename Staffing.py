import numpy as np
import pandas as pd
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# 부서 직렬 데이터
job_titles = [
    "AdministrativeOfficer",
    "TaxOfficer",
    "ITSpecialist",
    "SocialWelfareWorker",
    "IndustrialEngineer",
    "AgriculturalOfficer",
    "GreenSpaceManager",
    "PublicHealthWorker",
    "MedicalTechnician",
    "Nurse",
    "EnvironmentalOfficer",
    "FacilitiesManager",
    "Driver",
    "ManagementOperationsOfficer",
    "Researcher",
    "Advisor",
    "OtherPositions",
]

departments = [
    "PlanBudge",
    "Audit",
    "PublicRelation",
    "JobEconomy",
    "FutureStrategy",
    "EnergyNewIndustry",
    "EducationSupport",
    "AthleticsPromotion",
    "Tourism",
    "ArtCulture",
    "EnvironmentManagement",
    "CityBeautification",
    "ParkGreenspace",
    "CultureArtSpecializePlanUnit",
    "Construction",
    "SafetyDisaster",
    "City",
    "TransportationAdmin",
    "BuildinPpermit",
    "WaterSewer",
    "YeongsanpoDevelopPlanOrganization",
    "Secretary",
    "ResidentAssistance",
    "SocialService",
    "Taxation",
    "Accounting",
    "CivicService",
    "AgriculturePolicy",
    "HorticultureDistribution",
    "FoodPlan",
    "AgriculturePromotion",
    "TechnicalSupport",
    "AnimalHusbandry",
    "HealthAdmin",
    "InfectDiseaseManagement",
    "HealthWellness",
    "BitgaramFacilityManagementOffice",
]

# 부서 인원 분포 (각 직렬 별로 각 부서에 배치할 인원 수를 정의)
department_job_mapping = {
    "PlanBudge": [0.5, 0.2, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.1],
    "Audit": [0.3, 0.3, 0.1, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    "PublicRelation": [0.4, 0.2, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0.1],
    "JobEconomy": [0.6, 0, 0, 0.2, 0, 0.1, 0, 0.1, 0, 0, 0, 0.1],
    "FutureStrategy": [0.5, 0, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.1],
    "EnergyNewIndustry": [0.4, 0, 0.3, 0, 0, 0, 0, 0, 0, 0.1, 0, 0.2],
    "EducationSupport": [0.5, 0.1, 0.1, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    "AthleticsPromotion": [0.7, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.1],
    "Tourism": [0.6, 0, 0, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    "ArtCulture": [0.5, 0.1, 0, 0.3, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    "EnvironmentManagement": [0.4, 0, 0.1, 0, 0.1, 0.2, 0.1, 0.1, 0, 0, 0, 0.1],
    "CityBeautification": [0.4, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0.1],
    "ParkGreenspace": [0.3, 0, 0, 0.1, 0.1, 0.4, 0.1, 0, 0, 0, 0, 0.1],
    "CultureArtSpecializePlanUnit": [0.6, 0, 0, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    "Construction": [0.5, 0.1, 0.2, 0, 0.1, 0.1, 0, 0, 0, 0, 0, 0.1],
    "SafetyDisaster": [0.4, 0.1, 0.1, 0.2, 0.1, 0, 0, 0.1, 0, 0, 0, 0.1],
    "City": [0.6, 0.1, 0, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0],
    "TransportationAdmin": [0.5, 0.1, 0.1, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    "BuildinPpermit": [0.4, 0.1, 0.2, 0.1, 0, 0.1, 0, 0, 0, 0, 0, 0.1],
    "WaterSewer": [0.3, 0, 0, 0.1, 0.1, 0.2, 0.1, 0, 0, 0, 0, 0.1],
    "YeongsanpoDevelopPlanOrganization": [
        0.4,
        0.2,
        0.1,
        0,
        0.1,
        0,
        0.1,
        0,
        0,
        0,
        0,
        0.1,
    ],
    "Secretary": [0.5, 0.1, 0.1, 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    "ResidentAssistance": [0.4, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.1],
    "SocialService": [0.3, 0.1, 0, 0.4, 0, 0, 0, 0.1, 0, 0, 0, 0.1],
    "Taxation": [0.7, 0.2, 0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0],
    "Accounting": [0.6, 0.2, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0],
    "CivicService": [0.5, 0.2, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.1],
    "AgriculturePolicy": [0.3, 0, 0, 0.1, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0.1],
    "HorticultureDistribution": [0.3, 0.2, 0, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0],
    "FoodPlan": [0.5, 0, 0, 0.2, 0.1, 0, 0, 0.1, 0, 0, 0, 0.1],
    "AgriculturePromotion": [0.4, 0, 0, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0.1],
    "TechnicalSupport": [0.5, 0, 0.1, 0.2, 0, 0.2, 0, 0, 0, 0, 0, 0.1],
    "AnimalHusbandry": [0.4, 0, 0, 0.1, 0.2, 0.2, 0, 0, 0, 0, 0, 0.1],
    "HealthAdmin": [0.5, 0.1, 0.1, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.1],
    "InfectDiseaseManagement": [0.5, 0, 0, 0, 0, 0, 0.1, 0.2, 0, 0, 0, 0.1],
    "HealthWellness": [0.4, 0, 0, 0.3, 0, 0, 0.1, 0, 0, 0, 0, 0.1],
    "BitgaramFacilityManagementOffice": [0.3, 0, 0.1, 0, 0, 0, 0.1, 0.2, 0, 0, 0, 0.1],
}

# 각 직렬에 대한 배치 비율을 기반으로 성과 기여도를 설정 (가상의 값)
performance_contributions = {
    "AdministrativeOfficer": 0.8,
    "TaxOfficer": 1.0,
    "ITSpecialist": 0.9,
    "SocialWelfareWorker": 0.7,
    "IndustrialEngineer": 0.6,
    "AgriculturalOfficer": 0.5,
    "GreenSpaceManager": 0.4,
    "PublicHealthWorker": 0.9,
    "MedicalTechnician": 0.85,
    "Nurse": 0.75,
    "EnvironmentalOfficer": 0.7,
    "FacilitiesManager": 0.6,
    "Driver": 0.3,
    "ManagementOperationsOfficer": 0.5,
    "Researcher": 0.95,
    "Advisor": 0.8,
    "OtherPositions": 0.6,
}

# 각 직렬별 배치할 공무원 수 (가상의 데이터)
job_counts = [522, 29, 11, 131, 43, 58, 22, 47, 23, 32, 22, 144, 35, 9, 7, 33, 37]

# 선형 계획법 문제 정의
num_departments = len(departments)
num_jobs = len(job_titles)

for department, job_mapping in department_job_mapping.items():
    if len(job_mapping) < len(job_titles):
        department_job_mapping[department] = job_mapping + [0] * (
            len(job_titles) - len(job_mapping)
        )

# 목적 함수: 부서별 성과 기여도를 최대화하기 위해 각 직렬의 기여도를 배치 비율과 곱하여 최대화
c = np.zeros(num_jobs * num_departments)

for i, department in enumerate(departments):
    for j, job in enumerate(job_titles):
        c[i * num_jobs + j] = (
            performance_contributions[job] * department_job_mapping[department][j]
        )

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
result = linprog(
    c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
)

# 결과 확인
if result.success:
    print("최적화 성공")
    optimized_allocation = result.x.reshape((num_departments, num_jobs))
    print(optimized_allocation)

    # 최적화된 배치 결과를 pandas DataFrame으로 변환
    optimized_df = pd.DataFrame(
        optimized_allocation, index=departments, columns=job_titles
    )

    # CSV 파일로 저장
    optimized_df.to_csv("staffing_base.csv", encoding="utf-8-sig")  # CSV로 저장
else:
    print("최적화 실패")

# 결과 시각화 (배치된 공무원 수 시각화)
optimized_allocation = result.x.reshape((num_departments, num_jobs))
fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(optimized_allocation, cmap="viridis", aspect="auto")
ax.set_xticks(np.arange(len(job_titles)))
ax.set_yticks(np.arange(len(departments)))
ax.set_xticklabels(job_titles)
ax.set_yticklabels(departments)
plt.xticks(rotation=90)
plt.xlabel("Job Titles")
plt.ylabel("Departments")
plt.title("Optimized Allocation of Public Employees")
fig.colorbar(im)
plt.show()
