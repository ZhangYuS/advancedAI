import argparse, os, random, time
from shutil import copyfile

label_num = ['human', 'cat', 'dog']

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat_num', type=int, default=50, help='number of cat pictures')
    parser.add_argument('--human_num', type=int, default=300, help='number of human pictures')
    parser.add_argument('--dog_num', type=int, default=300, help='number of dog pictures')
    arg = parser.parse_args()
    return arg

def load_data(data_path):
    ret = []
    for phase in ['train', 'val']:
        data_file = open(os.path.join(data_path, phase, f'{phase}_file_list.txt'), 'r', encoding='utf8')
        data = [[x.split('\t')[0], int(x.split('\t')[1])] for x in data_file.readlines()]
        ret.append(data)
    return ret[0], ret[1]

def generate_data_file(train_data, val_data, arg):
    if [x[1] for x in train_data].count(label_num.index('cat')) < arg.cat_num:
        raise ValueError('total number of cat less than cat_num')
    if [x[1] for x in train_data].count(label_num.index('human')) < arg.human_num:
        raise ValueError('total number of human less than normal_num')
    if [x[1] for x in train_data].count(label_num.index('dog')) < arg.dog_num:
        raise ValueError('total number of dog less than normal_num')

    human_train_data = list(filter(lambda x: True if x[1] == label_num.index('human') else False, train_data))
    cat_train_data = list(filter(lambda x: True if x[1] == label_num.index('cat') else False, train_data))
    dog_train_data = list(filter(lambda x: True if x[1] == label_num.index('dog') else False, train_data))

    ret_train_data = random.choices(human_train_data, k=arg.human_num)\
                     + random.choices(cat_train_data, k=arg.cat_num)\
                     + random.choices(dog_train_data, k=arg.dog_num)
    ret_val_data = val_data
    return ret_train_data, ret_val_data

def make_dir(target_data_path, arg):
    if not os.path.exists(target_data_path):
        os.mkdir(target_data_path)
    date = time.strftime("%m-%d-%H-%M-%S")
    data_file_name = f'{date}_{arg.cat_num}_{arg.human_num}_{arg.dog_num}'
    os.mkdir(os.path.join(target_data_path, data_file_name))
    os.mkdir(os.path.join(target_data_path, data_file_name, 'train'))
    os.mkdir(os.path.join(target_data_path, data_file_name, 'val'))
    os.mkdir(os.path.join(target_data_path, data_file_name, 'train', 'processed'))
    os.mkdir(os.path.join(target_data_path, data_file_name, 'val', 'processed'))
    return os.path.join(target_data_path, data_file_name, 'train'), \
           os.path.join(target_data_path, data_file_name, 'val')

def copy_data(origin_path, target_path, file_name_list):
    for file_name in file_name_list:
        copyfile(os.path.join(origin_path, file_name), os.path.join(target_path, file_name))

def save_data(train_data, val_data, origin_data_path, target_data_path, arg):
    target_train_path, target_val_path = make_dir(target_data_path, arg)
    with open(os.path.join(target_train_path, 'train_file_list.txt'), 'w', encoding='utf8') as f:
        print('\n'.join([f'{x[0]}\t{x[1]}' for x in train_data]), file=f)
    with open(os.path.join(target_val_path, 'val_file_list.txt'), 'w', encoding='utf8') as f:
        print('\n'.join([f'{x[0]}\t{x[1]}' for x in val_data]), file=f)
    target_train_picture_path = os.path.join(target_train_path, 'processed')
    target_val_picture_path = os.path.join(target_val_path, 'processed')
    origin_train_picture_path = os.path.join(origin_data_path, 'train', 'processed')
    origin_val_picture_path = os.path.join(origin_data_path, 'val', 'processed')
    copy_data(origin_train_picture_path, target_train_picture_path, [x[0] for x in train_data])
    copy_data(origin_val_picture_path, target_val_picture_path, [x[0] for x in val_data])




def main():
    arg = parse_arg()
    origin_data_path = 'data'
    target_data_path = 'unbalanced_data'
    train_data, val_data = load_data(origin_data_path)
    train_data, val_data = generate_data_file(train_data, val_data, arg)
    save_data(train_data, val_data, origin_data_path, target_data_path, arg)

if __name__ == '__main__':
    main()
