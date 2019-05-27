import os

filename2score = lambda x: x[:x.rfind('.')].split('_')[-2:]

filenames = os.listdir('dataset/images')

with open('dataset/train.txt', 'w') as f_train_txt:
    for filename in filenames[:50000]:
        fx, fy = filename2score(filename)
        line = '{} {} {}\n'.format(filename, fx, fy)
        f_train_txt.write(line)

with open('dataset/val.txt', 'w') as f_val_txt:
    for filename in filenames[50000:60000]:
        fx, fy = filename2score(filename)
        line = '{} {} {}\n'.format(filename, fx, fy)
        f_val_txt.write(line)

with open('dataset/test.txt', 'w') as f_test_txt:
    for filename in filenames[60000:]:
        line = '{}\n'.format(filename)
        f_test_txt.write(line)
