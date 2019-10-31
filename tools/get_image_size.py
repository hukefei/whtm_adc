import os
import pandas as pd
import json

size_file = r'C:\Users\huker\Desktop\size.csv'
image_list = r'C:\Users\huker\Desktop\1GE02_img_list.csv'
json_file = r'C:\Users\huker\Desktop\1GE02_img_size.json'

size_df = pd.read_csv(size_file)
img_df = pd.read_csv(image_list)

joined = pd.merge(size_df, img_df, 'inner', left_on='images_name', right_on='image_name')
print('merged {} images from {}'.format(len(joined), len(img_df)))

size_dict = {}
for i in joined.index:
    img_name = joined.loc[i, 'image_name']
    size = joined.loc[i, 'size']
    size_dict[img_name] = str(size)

print(size_dict)
with open(json_file, 'w') as f:
    json.dump(size_dict, f, indent=4)