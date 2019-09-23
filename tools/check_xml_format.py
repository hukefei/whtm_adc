import os
import re
import shutil

file_path = r'F:\XMTM\station\14103\data\14103_0819'

site_code = os.listdir(file_path)

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
                        print('xml content error: {} has no {}'.format(os.path.join(root, file), code))
                if not os.path.exists(os.path.join(root, file.replace('xml', 'JPG'))):
                    if not os.path.exists(os.path.join(root, file.replace('xml', 'jpg'))):
                        print('{} has no corresponding jpg file'.format(file))
                        os.unlink(os.path.join(root, file))

