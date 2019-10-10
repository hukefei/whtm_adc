import os
import xml.etree.ElementTree as ET

defect_path = r'D:\Project\WHTM\data\21101\modify_code'
xml_path = r'D:\Project\WHTM\data\21101\final_dataset\annotations'

for root, _, files in os.walk(defect_path):
    code_root = root.replace('/', '\\').split('\\')[-1]
    print(code_root)
    print('-'*50)
    for file in files:
        file_name = file.split('__')[0]
        code = file.split('__')[1]
        xmin = file.split('__')[2]
        ymin = file.split('__')[3]
        xmax = file.split('__')[4]
        ymax = file.split('__')[5].split('.')[0]
        print(file_name, code, xmin, ymin, xmax, ymax)
        xml_name = file_name + '.xml'
        xml = os.path.join(xml_path, xml_name)
        if not os.path.exists(xml):
            print('{} not exists'.format(xml))
            continue
        tree = ET.parse(xml)
        tree_root = tree.getroot()
        objs = tree_root.findall('object')
        for obj in objs:
            category = obj[0].text
            bbox = [int(obj[4][i].text) for i in range(4)]
            if bbox == [int(xmin), int(ymin), int(xmax), int(ymax)]:
                obj[0].text = code_root
                print('changed')
        tree.write(xml)

