import os,json
from PIL import Image
import random
true_image_root='../autodl-tmp/test'
true_image_list=os.listdir(true_image_root)
path_pool=[]
for image_name in true_image_list:
    path_pool.append((os.path.join(true_image_root,image_name),1))
with open('./image_list.json','r') as f:
    norm_split=json.load(f)['test']
select_path=[]
for image_path in norm_split:
    if not os.path.basename(image_path).startswith("re_"):
        select_path.append((image_path,0))
true_image_number=len(path_pool)
random.shuffle(select_path)
path_pool=path_pool+select_path[:true_image_number]
total_number=len(path_pool)
split_number1=int(total_number*0.5)
split_number2=int(total_number*0.75)
random.shuffle(path_pool)
res={
    'train':path_pool[:split_number1],
    'val':path_pool[split_number1+1:split_number2],
    'test':path_pool[split_number2+1:]
}
with open('./fintune.json','w') as f:
    json.dump(res,f)