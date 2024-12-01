import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# 부서와 구성원 수를 포함한 CSV 파일 로드
members_df = pd.read_csv('./employees_per_department.csv', header=None, encoding='ascii')
members_df.columns = ['number', 'num_employees']
members_df['number'] = pd.to_numeric(members_df['number'], errors='coerce', downcast='integer')
members_df['num_employees'] = pd.to_numeric(members_df['num_employees'], errors='coerce')
