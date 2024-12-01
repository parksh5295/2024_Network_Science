import chardet
import pandas as pd
filename = "./employees_per_department.csv"
with open(filename, 'rb') as f:
    result = chardet.detect(f.readline())  # or read() if the file is small.
    print(result['encoding'])