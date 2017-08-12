
from PIL import Image
import os, sys

def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.
    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.
      num_train: number of training data
    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    # print 'DATASET DIR:', dataset_dir
    # print 'subdir:', [name for name in os.listdir(dataset_dir)]
    # dataset_main_folder_list = []
    # for name in os.listdir(dataset_dir):
    # 	if os.path.isdir(name):
    # 		dataset_main_folder_list.append(name)
    dataset_main_folder_list = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,name))]
    dataset_root = os.path.join(dataset_dir, dataset_main_folder_list[0])
    directories = []
    class_names = []
    for filename in os.listdir(dataset_root):
        path = os.path.join(dataset_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
  
    count = 0
    #print(directories)
    for directory in directories:
        #print(directory)
        #continue
        for filename in os.listdir(directory):
            print(filename)
            path = os.path.join(directory, filename)

            im = Image.open(path)
            imResize = im.resize((28,28), Image.ANTIALIAS)
            imResize.save(path, 'bmp')
            print(count)
            count = count + 1
            


    
    return
    
def main():
    _get_filenames_and_classes('/notebooks/mnist_prac/3_convet_keras/')
    
    #print(len(photo_filenames))
    #print(class_names)
    
    
if __name__ == "__main__":
    main()
    
    