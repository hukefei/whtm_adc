import os
import re
import shutil

file_path = r'D:\Project\WHTM\data\21101\annotations'

site_code = [ 'AZ08', 'AZ19', 'COM01', 'COM03', 'COM15', 'COM99', 'PLN01',
             'REP01', 'RES03', 'RES06', 'STR02', 'STR04', 'QS']

for root, dirs, files in os.walk(file_path):
    for file in files:
        if '.xml' in file:
            with open(os.path.join(root, file), 'r', encoding= 'utf-8') as f:
                content = f.read()
                codes_in_xml = re.findall('<name>(.*?)</name>', content)
            if len(codes_in_xml) == 0:
                print('{} is empty'.format(file))
                os.unlink(os.path.join(root, file))
            else:
                for code in codes_in_xml:
                    if code not in site_code:
                        print('xml content error: {} has wrong code: {}'.format(os.path.join(root, file), code))

