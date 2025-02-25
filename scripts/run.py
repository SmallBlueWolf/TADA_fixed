import subprocess
from tqdm import tqdm
import yaml

# 定义读取文件并执行的函数
def process_file(file_path, config_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Processing prompts", position=1):
        line = line.strip()
        if line:
            subprocess.run(['/home/bluewolf/miniconda3/envs/tada/bin/python', '-m', 'apps.run', '--config', config_path, '--text', line])

def update_yaml_field(file_path, new_workspace):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if 'training' in config:
        config['training']['workspace'] = new_workspace
    else:
        raise Exception("未找到 'training' 节点")

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)

if __name__ == '__main__':
    file_paths = ['./prompts/celebrities.txt',
                 './prompts/FC.txt',
                 './prompts/GJD.txt',
                 './prompts/IOVT.txt']
    config_path = './configs/tada.yaml'
    for file_path in tqdm(file_paths, total=len(file_paths), desc="Four kinds of propmts files", position=0):
        new_workspace = file_path.split('/')[-1].split('.')[0]
        update_yaml_field(config_path, new_workspace)
        process_file(file_path, config_path)