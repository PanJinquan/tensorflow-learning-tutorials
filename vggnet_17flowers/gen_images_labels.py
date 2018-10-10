# coding: utf-8
import os
import os.path
import TxtStorage as txts

def get_files_list(dir):
    '''
    实现遍历dir目录下,所有文件(包含子文件夹的文件)
    :param dir:指定文件夹目录
    :return:包含所有文件的列表->list
    '''
    # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
    files_list=[]
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            # print("parent is: " + parent)
            # print("filename is: " + filename)
            # print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
            files_list.append([os.path.join(parent, filename)])
    return files_list

def get_files_labels(dir):
    files_list=[]
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            # print("parent is: " + parent)
            # print("filename is: " + filename)
            # print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
            labels=parent.split('/')[-1]
            path=os.path.join(parent, filename)

            files_list.append([path,int(labels)])
    return files_list

if __name__=='__main__':

    dir = '17flowers'
    txt_filename='imagelist.txt'
    files_list=get_files_labels(dir)
    for l in files_list:
        print(l)

    txt_str = txts.TxtStorage()
    txt_str.save_txt(files_list, txt_filename, mode='w')
