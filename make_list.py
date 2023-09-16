import os
def write_name(np, tx):
    #npz文件路径
    files = os.listdir(np)
    #txt文件路径
    f = open(tx, 'w')
    for i in files:
        #name = i.split('\\')[-1]
        name = i[:-4]+'\n'
        f.write(name)
        
write_name('./data/Synapse/train_npz', './lists/lists_Synapse/train.txt')
#write_name('./data/Synapse/test_vol_h5', './lists/lists_Synapse/test_vol.txt')