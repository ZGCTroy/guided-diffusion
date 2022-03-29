import os

dir_path = '/mnt/data1/shengming/tiny-imagenet-200/train'

dir_list = os.listdir(dir_path)
print(len(dir_list))

# f = open('list_tiny_imagenet.txt',mode='w') 
# for dir_name in dir_list:
#     f.write(f'{dir_name}\n')

# print(f.readlines())
with open('list_tiny_imagenet.txt', "r") as tiny_file:
    # tiny_fi
    tiny_file = tiny_file.readlines()
    tiny_file = [p.strip() for p in tiny_file]
    # Attr_info = Attr_info[1:]

    print(tiny_file)