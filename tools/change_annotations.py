import os
import xml.etree.ElementTree as ET
xml_dir = r'/data/sdv1/whtm/mask/data/trainval_0715/val'

for root, _, files in os.walk(xml_dir):
    for file in files:
        if file.endswith('xml'):
            xml_file = os.path.join(root, file)
            tree = ET.parse(xml_file)
            tree_root = tree.getroot()
            objs = tree_root.findall('object')
            for obj in objs:
                if obj[0].text == 'LS1':
                    obj[0].text = 'LA03'
                    print(file)
            tree.write(xml_file)