import pandas as pd

def excel_to_txt(excel_file, txt_file):
    # 读取Excel文件
    df = pd.read_excel(excel_file)

    # 将数据保存为文本文件
    df.to_csv(txt_file, sep='\t', index=False, header=None)

# 示例用法
excel_file = '/home/intuser/Desktop/lyy/capstone-master/text_gcn/data/input.xlsx'
txt_file = '/home/intuser/Desktop/lyy/capstone-master/text_gcn/data/input.txt'

excel_to_txt(excel_file, txt_file)
