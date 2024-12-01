import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np

# 부서 목록
departments = [
    "PlanBudge", "Audit", "PublicRelation", "JobEconomy", "FutureStrategy", "EnergyNewIndustry", 
    "EducationSupport", "AthleticsPromotion", "Tourism", "ArtCulture", "EnvironmentManagement",
    "CityBeautification", "ParkGreenspace", "CultureArtSpecializePlanUnit", "Construction", 
    "SafetyDisaster", "City", "TransportationAdmin", "BuildinPpermit", "WaterSewer",
    "YeongsanpoDevelopPlanOrganization", "Secretary", "ResidentAssistance", "SocialService", 
    "Taxation", "Accounting", "CivicService", "AgriculturePolicy", "HorticultureDistribution", 
    "FoodPlan", "AgriculturePromotion", "TechnicalSupport", "AnimalHusbandry", "HealthAdmin",
    "InfectDiseaseManagement", "HealthWellness", "BitgaramFacilityManagementOffice"
]

# 직렬 목록
job_titles = [
    "AdministrativeOfficer", "TaxOfficer", "ITSpecialist", "SocialWelfareWorker", 
    "IndustrialEngineer", "AgriculturalOfficer", "GreenSpaceManager", "PublicHealthWorker", 
    "MedicalTechnician", "Nurse", "EnvironmentalOfficer", "FacilitiesManager", 
    "Driver", "ManagementOperationsOfficer", "Researcher", "Advisor", "OtherPositions"
]

# 직렬별 인원 수
job_counts = np.array([
    522, 29, 11, 131, 43, 58, 22, 47, 23, 32, 22, 144, 35, 9, 7, 33, 37
])

# 부서별 직렬 배치 비율 (기존 비율 유지)
department_job_mapping = {
    "PlanBudge": [0.05, 0.1, 0.05, 0, 0, 0, 0, 0.05, 0.05, 0, 0.05, 0, 0, 0.05, 0.1, 0.05, 0],
    "Audit": [0.1, 0.15, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0, 0.05, 0.05, 0.05, 0.1, 0.05, 0],
    # 나머지 부서도 추가하세요
}

# 부서별 직렬 배치 수 계산
def assign_staff_to_departments(total_staff, job_counts, department_job_mapping):
    total_jobs = sum(job_counts)
    department_staff = {department: [0] * len(job_counts) for department in department_job_mapping}

    for department, job_ratios in department_job_mapping.items():
        department_total = total_staff * sum(job_ratios) / len(department_job_mapping)

        # 직렬별 고용 인원 수 분배
        for i in range(len(job_counts)):
            department_staff[department][i] = round(department_total * job_ratios[i])

    # 직렬별 배치 수가 정확히 job_counts와 일치하도록 최적화
    department_staff_flat = [sum([department_staff[dept][i] for dept in department_staff]) for i in range(len(job_counts))]
    
    diff = [job_counts[i] - department_staff_flat[i] for i in range(len(job_counts))]
    
    for i in range(len(job_counts)):
        for dept in department_staff:
            if diff[i] != 0:
                department_staff[dept][i] += diff[i]
    
    return department_staff

# 직렬별 배치 수 계산
department_staff = assign_staff_to_departments(1205, job_counts, department_job_mapping)

# 시각화
G = nx.Graph()

# 직렬 노드 추가
for serial in job_titles:
    G.add_node(serial, bipartite=0)

# 부서 노드 추가
for department in departments:
    G.add_node(department, bipartite=1)

# 부서와 직렬을 연결
for department, ratios in department_job_mapping.items():
    for i, ratio in enumerate(ratios):
        if ratio > 0:
            G.add_edge(job_titles[i], department, weight=ratio)

pos = nx.spring_layout(G, k=6, seed=42)
# 직렬 노드는 왼쪽, 부서 노드는 오른쪽에 배치
for i, serial in enumerate(job_titles):
    pos[serial] = (0, i)
for i, department in enumerate(departments):
    pos[department] = (1, i)

plt.figure(figsize=(12, 12))
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")
plt.title("Bipartite Network: Departments and Serial Jobs")
plt.show()

# 부서별로 총 인원 수를 계산
output_data = []

# 부서별로 총 인원 수를 계산
for department, staff in department_staff.items():
    total_members = sum(staff)
    output_data.append({
        "department_number": len(output_data) + 1,  # 1부터 시작하는 부서 번호
        "department_id": department,
        "member_number": total_members
    })

# DataFrame으로 변환
df = pd.DataFrame(output_data)

# CSV로 저장
df.to_csv("./staffing_base_total.csv", index=False)
