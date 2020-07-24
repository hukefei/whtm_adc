import os
import re
import shutil

def check_annotations(file_path, codes=[]):
    """
    check if the xml annotation has no bbox or has wrong codes.
    annotation with no bbox will be removed.
    annotation with wrong code will be highlighted and should be modified manually.
    :param file_path: file of annotations.
    :param codes: available codes.
    :return:
    """
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if '.xml' in file:
                print(file)
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    codes_in_xml = re.findall('<name>(.*?)</name>', content)
                if len(codes_in_xml) == 0:
                    print('{} is empty'.format(file))
                    os.unlink(os.path.join(root, file))
                elif codes != []:
                    for code in codes_in_xml:
                        if code not in codes:
                            print('xml content error: {} has wrong code: {}'.format(os.path.join(root, file), code))
                else:
                    pass

if __name__ == '__main__':
    file_path = r'/data/sdv1/whtm/56A02/data/annotations'
    code_file = r'/data/sdv1/whtm/56A02/data/classes.txt'
    with open(code_file) as f:
        site_code = f.read().splitlines()
    check_annotations(file_path, site_code)
