#! /usr/bin/env python
import gc
import argparse
import numpy as np


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def count_line_num(sl_filepath, fheader=True):
    count = 0
    with open(sl_filepath, 'r') as rf:
        if fheader:
            next(rf)
        for line in rf:
            count += 1
    print('done count the lines of file..')
    return count


def read_one_shuffle_info(filepath, shuffle_lines_num, total_lines_num, checked_lines_num, isheader):
    with open(filepath, 'r') as rf:
        if isheader:
            next(rf)
        count = 0
        while count < checked_lines_num:
            next(rf)
            count += 1

        count = 0
        lines_info = []
        lines_num = min(shuffle_lines_num, (total_lines_num - checked_lines_num))
        for line in rf:
            if count < lines_num:
                lines_info.append(line.strip())
                count += 1
            else:
                break
        print('done reading file {}'.format(filepath))
        return lines_info


def shuffle_samples(samples_info):
    mark = list(range(len(samples_info)))
    np.random.shuffle(mark)
    shuffled_samples = []
    for i in mark:
        shuffled_samples.append(samples_info[i])
    return shuffled_samples


def write_to_one_file_append(features_info, wfilepath):
    with open(wfilepath, 'a') as wf:
        for i in range(0, len(features_info)):
            wf.write(features_info[i] + '\n')
    print('done writing features info to {}'.format(wfilepath))


def caoncat_two_files(file1, file2, shuffle_lines_num, lines_num, concated_fp, isheader):
    open(concated_fp, 'w').close()

    if isheader:
        rf1 = open(file1, 'r')
        wf = open(concated_fp, 'a')
        wf.write(next(rf1))
        wf.close()
        rf1.close()

    f1line_count = count_line_num(file1, isheader)
    f2line_count = count_line_num(file2, isheader)

    line_ratio = float(f2line_count) / f1line_count
    shuffle_lines_num2 = round(line_ratio * shuffle_lines_num) + 1

    checked_lines_num1, checked_lines_num2 = 0, 0
    while checked_lines_num1 < lines_num or checked_lines_num2 < lines_num:
        file1_info = read_one_shuffle_info(file1, shuffle_lines_num, lines_num, checked_lines_num1, isheader)
        checked_lines_num1 += len(file1_info)
        file2_info = read_one_shuffle_info(file2, shuffle_lines_num2, lines_num, checked_lines_num2, isheader)
        checked_lines_num2 += len(file2_info)
        if len(file1_info) == 0 and len(file2_info) == 0:
            break
        samples_info = shuffle_samples(file1_info + file2_info)
        write_to_one_file_append(samples_info, concated_fp)

        del file1_info
        del file2_info
        del samples_info
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='concat two LINE-files with shuffling the rows')
    parser.add_argument('--fp1', type=str, required=True,
                        help='file path1')
    parser.add_argument('--fp2', type=str, required=True,
                        help='file path2')
    parser.add_argument('--concated_fp', type=str, required=True,
                        help='concated file path')

    parser.add_argument('--num_samples_per_file', type=int, default=1000000000, required=False,
                        help='num of samples per file, default 1000000000 (equal to Inf)')
    parser.add_argument('--num_lines_shuffle', type=int, default=2000000, required=False,
                        help='num of lines for one shuffle, default 2000000')
    parser.add_argument('--header', type=str, default='no', required=False,
                        help='whether there are headers in fp1 and fp2 or not')
    args = parser.parse_args()

    filepath1 = args.fp1
    filepath2 = args.fp2
    concated_filepath = args.concated_fp
    linenum = args.num_samples_per_file
    oneshufflenum = args.num_lines_shuffle
    header = str2bool(args.header)

    caoncat_two_files(filepath1, filepath2, oneshufflenum, linenum, concated_filepath, header)
    print('done writing the data..')
