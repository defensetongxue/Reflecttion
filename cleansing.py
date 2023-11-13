import os,json
from tools.mirror import mirror
import random
data_root='../autodl-tmp/imagenet100'
folder_dir=os.listdir('../autodl-tmp/imagenet100')
image_path_list=[]
for image_folder in folder_dir:
    image_list=os.listdir(os.path.join(data_root,image_folder))
    for image_name in folder_dir:
        mirror(os.path.join(data_root,image_folder,image_name),
               os.path.join(data_root,image_folder,"re_"+image_name))
        image_path_list.append(os.path.join(data_root,image_folder,image_name))
        image_path_list.append(os.path.join(data_root,image_folder,"re_"+image_name))
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