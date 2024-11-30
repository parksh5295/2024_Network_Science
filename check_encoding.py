import chardet
import pandas as pd
filename = "./Low_Data/Department_AnalyzeScorecardResult/file1.csv"
with open(filename, 'rb') as f:
    result = chardet.detect(f.readline())  # or read() if the file is small.
    print(result['encoding'])