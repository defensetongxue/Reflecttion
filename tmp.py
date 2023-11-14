import json,os
with open('./image_list.json','r') as f:
    res=json.load(f)
for split in res:
    split_list=res[split]
    
    positive_CNT=0
    negtive_cnt=0
    for image_path in split_list:
        image_name=os.path.basename(image_path)
        if image_name.startswith('re_'):
            positive_CNT+=1
        else:
            negtive_cnt+=1
    print(f"{split} reflection: {str(positive_CNT)} normal {str(negtive_cnt)}")