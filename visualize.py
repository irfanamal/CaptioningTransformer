from matplotlib import pyplot as plt

train_losses = [2000, 3000, 4000, 5000, 6000, 7000]
val_losses = [2.146, 2.113, 2.151, 2.046, 2.192, 2.156]
# with open('dataset/clean_semantics.txt') as f:
#     lines = f.readlines()
# for line in lines:
    # elements = line.split(',')
    # train_loss = float(elements[1].split()[-1])
    # val_loss = float(elements[2].split()[-1])
    # if len(train_losses) < 40:
    #     train_losses.append(train_loss)
    #     val_losses.append(val_loss)
    # else:
    #     break
    # elements = line.split()
    # idf_value = float(elements[1])
    # if idf_value not in train_losses:
    #     train_losses.append(idf_value)
    #     val_losses.append(1)
    # else:
    #     val_losses[train_losses.index(idf_value)] += 1

plt.plot(train_losses, val_losses)
# plt.plot([i for i in range(1,len(val_losses)+1)], val_losses, label='Val Loss')
plt.ylim(0, 10)
plt.xlabel('Feed Forward Dimensions')
plt.ylabel('CIDER')
# plt.legend()
plt.show()