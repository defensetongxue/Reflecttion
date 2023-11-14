import os,json
from tools.mirror import mirror
import random
from tqdm import tqdm
data_root='../autodl-tmp/nat'
folder_dir=os.listdir(data_root)
total_folder=len(folder_dir)
res={'train':[],'val':[],'test':[]}
map={
    "seg_pred":'val',  "seg_test":'test',  "seg_train":'train'
}
for split_folder in folder_dir:
    
    class_folders=os.listdir(os.path.join(data_root,split_folder))
    split_name =map[split_folder]
    for class_folder in class_folders:
        image_list = os.listdir(os.path.join(data_root,split_folder,class_folder))
        for image_name in image_list: 
            if image_name.startswith("re_"):
                continue
            image_path=os.path.join(data_root,split_folder,class_folder,image_name)
            mirror_path=os.path.join(data_root,split_folder,class_folder,"re_"+image_name)
            res[split_name].append(image_path)
            res[split_name].append(mirror_path)
            if os.path.exists(mirror_path):
                continue
            mirror(image_path,
                   mirror_path)
with open('./image_list.json','w') as f:
    json.dump(res,f)