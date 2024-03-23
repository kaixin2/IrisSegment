import os

file_path = "Data/CAV/guassian_noise_224/train/img\S1001L07.jpg"

# 获取文件名
file_name = os.path.basename(file_path)

# 使用split方法分割文件名，以'.'为分隔符，取第一部分
desired_part = file_name.split('.')[0] + '.'

print(desired_part)
