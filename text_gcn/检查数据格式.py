def check_format(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()

    invalid_lines = []

    for i, line in enumerate(lines):
        line = line.strip()
        if '\t' not in line:
            invalid_lines.append(i+1)  # 添加行号到无效行列表

    if invalid_lines:
        print("Invalid format detected in the following lines:")
        for line_num in invalid_lines:
            print(f"Line {line_num}")
    else:
        print("All lines have valid format.")

# 示例用法
txt_file = '/home/intuser/Desktop/lyy/capstone-master/text_gcn/data/output.txt'
check_format(txt_file)
