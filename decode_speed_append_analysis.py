import os

for i in range(100):
    file_name = f'gpu_sh{i}.sh'
    results = []
    if not os.path.exists(file_name):
        continue
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if 'run' in line:
                line = line.replace('.py', '_speed_test.py')
                line = line.replace('run', 'decode')
                results.append(line)
            else:
                results.append(line)
    with open(file_name, 'w') as f:
        for line in results:
            f.write(line + '\n')
