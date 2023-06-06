# with open('logs/CaT/best/all_test.txt', 'r') as f:
#     lines_transformer = f.readlines()
# with open('logs/CaT/sem/all_test.txt', 'r') as f:
#     lines_transformer_sem = f.readlines()
# with open('logs/NIC4Product/best/all_test.txt', 'r') as f:
#     lines_base = f.readlines()    
# with open('logs/NIC4Product/semantics/th5/rev3/all_test.txt', 'r') as f:
#     lines_base_sem = f.readlines()

# gt_len = []
# for i in range(1, len(lines_transformer), 10):
#     line = lines_transformer[i].split('Ground Truth: ')[-1][2:-3].split()
#     if len(line) not in gt_len:
#         gt_len.append(len(line))
#     # if len(line) == 11:
#     #     print(line)


# count = [0 for _ in range(max(gt_len)+1)]
# cider_transformer = [0 for _ in range(max(gt_len)+1)]
# cider_transformer_sem = [0 for _ in range(max(gt_len)+1)]
# cider_base = [0 for _ in range(max(gt_len)+1)]
# cider_base_sem = [0 for _ in range(max(gt_len)+1)]

# for i in range(0, len(lines_transformer), 10):
#     length = len(lines_transformer[i+1].split('Ground Truth: ')[-1][2:-3].split())
#     count[length] += 1
#     cider_transformer[length] += float(lines_transformer[i+7].split()[-1]) 
#     cider_transformer_sem[length] += float(lines_transformer_sem[i+7].split()[-1]) 
#     cider_base[length] += float(lines_base[i+7].split()[-1]) 
#     cider_base_sem[length] += float(lines_base_sem[i+7].split()[-1]) 

# avg_transformer = []
# avg_transformer_sem = []
# avg_base = []
# avg_base_sem = []
# for i, number in enumerate(count):
#     if number > 0:
#         avg_transformer.append(cider_transformer[i]/number)
#         avg_transformer_sem.append(cider_transformer_sem[i]/number)
#         avg_base.append(cider_base[i]/number)
#         avg_base_sem.append(cider_base_sem[i]/number)
#     else:
#         avg_transformer.append(0)
#         avg_transformer_sem.append(0)
#         avg_base.append(0)
#         avg_base_sem.append(0)

from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('dataset/train.csv')
count = [0 for _ in range(37)]
for i, produk in df['produk'].iteritems():
    produk = produk.split()
    count[len(produk)+2] += 1

# plt.plot([i for i in range(len(count))], avg_transformer, label='Transformer')
# plt.plot([i for i in range(len(count))], avg_transformer_sem, label='Transformer + Semantic')
# plt.plot([i for i in range(len(count))], avg_base, label='Baseline')
# plt.plot([i for i in range(len(count))], avg_base_sem, label='Baseline + Semantic')
plt.plot([i for i in range(len(count))], count)
plt.xlabel('Ground Truth Length')
plt.ylabel('Count')
# plt.legend()
plt.show()