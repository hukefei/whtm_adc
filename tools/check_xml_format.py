import os
import re
import shutil

file_path = r'F:\WHTM\MASK\data\AMI200\annotations'

code_file = r'F:\WHTM\24000\24000\data\train_val\classes.txt'
with open(code_file) as f:
    site_code = f.read().splitlines()

for root, dirs, files in os.walk(file_path):
    for file in files:
        if '.xml' in file:
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                content = f.read()
                codes_in_xml = re.findall('<name>(.*?)</name>', content)
            if len(codes_in_xml) == 0:
                print('{} is empty'.format(file))
                os.unlink(os.path.join(root, file))
            else:
                for code in codes_in_xml:
                    if code not in site_code:
                        print('xml content error: {} has wrong code: {}'.format(os.path.join(root, file), code))
