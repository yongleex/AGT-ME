import os


# Get all file names under a dir root
def get_all_files(root):
    output = []
    for roots, dir, files in os.walk(root, followlinks=True):
        for short_name in files:
            output = output + [os.path.join(roots, short_name)[len(root) + 1:]]
    return output


# delete all file names under a dir path
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
