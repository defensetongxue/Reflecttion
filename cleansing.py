import os,json
from tools.mirror import mirror
import random
from tqdm import tqdm
data_root='../autodl-tmp/imagenet100'
folder_dir=os.listdir('../autodl-tmp/imagenet100')
image_path_list=[]
cnt=0
total_folder=len(folder_dir)

for image_folder in folder_dir:
    
    image_list=os.listdir(os.path.join(data_root,image_folder))
    for image_name in image_list: 
        if image_name.startswith("re_"):
            continue
        image_path=os.path.join(data_root,image_folder,image_name)
        mirror_path=os.path.join(data_root,image_folder,"re_"+image_name)
        image_path_list.append(image_path)
        image_path_list.append(mirror_path)
        if os.path.exists(mirror_path):
            continue
        mirror(image_path,
               mirror_path)
    cnt+=1
    if cnt%10==0:
        print(f"finished {str(cnt)}/{str(total_folder)}")
random.shuffle(image_path_list)
total_number=len(image_path_list)
print("there is {} data".format(total_number))
res={
    'train':image_path_list[:int(total_number*0.5)],
    'val':image_path_list[int(total_number*0.5)+1:int(total_number*0.75)],
    'test':image_path_list[int(total_number*0.75)+1:]
}
with open('./image_list.json','w') as f:
    json.dump(res,f)