import os
import shutil


def del_file(pathdata):
    for x in os.listdir(pathdata):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = pathdata + "\\" + x  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data):  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)


def copy_demo(src_dir, dst_dir):
    """
    复制src_dir目录下的所有内容到dst_dir目录
    :param src_dir: 源文件目录
    :param dst_dir: 目标目录
    :return:
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if os.path.exists(src_dir):
        for file in os.listdir(src_dir):
            file_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file)
            if os.path.isfile(os.path.join(src_dir, file)):
                shutil.copy(file_path, dst_path)
            else:
                copy_demo(file_path, dst_path)
                print("存在多级文件夹，正在复制。")


pre = 1000

path = r'D:\opencv\perseonaldataset\standing'
original_name = os.listdir(path)
ids = 1
for i in original_name:
    print(i)
    text = str(i)
    newname = "0." + str(ids + pre) + ".jpg"
    os.rename(os.path.join(path, i), os.path.join(path, newname))
    ids = ids + 1
print("standing改名完成")

path = r'D:\opencv\perseonaldataset\falling'
original_name = os.listdir(path)
ids = 1
for i in original_name:
    print(i)
    text = str(i)
    newname = "1." + str(ids + pre) + ".jpg"
    os.rename(os.path.join(path, i), os.path.join(path, newname))
    ids = ids + 1
print("falling改名完成")

path = r'D:\opencv\perseonaldataset\sitting'
original_name = os.listdir(path)
ids = 1
for i in original_name:
    print(i)
    text = str(i)
    newname = "2." + str(ids + pre) + ".jpg"
    os.rename(os.path.join(path, i), os.path.join(path, newname))
    ids = ids + 1
print("sitting改名完成")


pre = 0

path = r'D:\opencv\perseonaldataset\standing'
original_name = os.listdir(path)
ids = 1
for i in original_name:
    print(i)
    text = str(i)
    newname = "0." + str(ids + pre) + ".jpg"
    os.rename(os.path.join(path, i), os.path.join(path, newname))
    ids = ids + 1
print("standing改名完成")

path = r'D:\opencv\perseonaldataset\falling'
original_name = os.listdir(path)
ids = 1
for i in original_name:
    print(i)
    text = str(i)
    newname = "1." + str(ids + pre) + ".jpg"
    os.rename(os.path.join(path, i), os.path.join(path, newname))
    ids = ids + 1
print("falling改名完成")

path = r'D:\opencv\perseonaldataset\sitting'
original_name = os.listdir(path)
ids = 1
for i in original_name:
    print(i)
    text = str(i)
    newname = "2." + str(ids + pre) + ".jpg"
    os.rename(os.path.join(path, i), os.path.join(path, newname))
    ids = ids + 1
print("sitting改名完成")

del_file(r'D:\opencv\perseonaldataset\mixed')

copy_demo(r'D:\opencv\perseonaldataset\standing', r'D:\opencv\perseonaldataset\mixed')
copy_demo(r'D:\opencv\perseonaldataset\falling', r'D:\opencv\perseonaldataset\mixed')
copy_demo(r'D:\opencv\perseonaldataset\sitting', r'D:\opencv\perseonaldataset\mixed')
