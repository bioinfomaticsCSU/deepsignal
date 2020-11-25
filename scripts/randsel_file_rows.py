#! /usr/bin/python
"""
select rows of file randomly, with a header
"""
import random
import argparse


def str2bool(v):
    # susendberg's function
    return v.lower() in ("yes", "true", "t", "1")


def random_select_file_rows(ori_file, w_file, maxrownum=10000, header=True):
    """

    :param ori_file:
    :param w_file:
    :param maxrownum:
    :param header:
    :return:
    """
    # whole_rows = open(ori_file).readlines()
    # nrows = len(whole_rows) - 1

    nrows = 0
    with open(ori_file) as rf:
        for line in rf:
            nrows += 1
    if header:
        nrows -= 1
    print('thera are {} lines (rm header if a header exists) in the file {}'.format(nrows, ori_file))

    actual_nline = maxrownum
    if nrows <= actual_nline:
        actual_nline = nrows
        print('gonna return all lines in ori_file {}'.format(ori_file))

    random_lines = random.sample(range(1, nrows+1), actual_nline)
    random_lines = [0] + sorted(random_lines)
    random_lines[-1] = nrows

    wf = open(w_file, 'w')
    with open(ori_file) as rf:
        if header:
            wf.write(next(rf))
        for i in range(1, len(random_lines)):
            chosen_line = ''
            for j in range(0, random_lines[i]-random_lines[i-1]):
                # print(j)
                chosen_line = next(rf)
            wf.write(chosen_line)
    wf.close()
    print('random_select_file_rows finished..')


def main():
    parser = argparse.ArgumentParser(description='select certain number of lines from a file randomly')
    parser.add_argument('--ori_filepath', type=str, required=True,
                        help='the path of file where lines are gonna be selected')
    parser.add_argument('--write_filepath', type=str, required=True,
                        help='the write filepath')
    parser.add_argument('--num_lines', type=int, required=True)
    parser.add_argument('--header', type=str, required=False,
                        default='true',
                        help='if the ori file has header or not. default true, t, yes, 1')

    args = parser.parse_args()

    orifile = args.ori_filepath
    wfile = args.write_filepath
    srownum = args.num_lines
    header = str2bool(args.header)
    random_select_file_rows(orifile, wfile, srownum, header)


if __name__ == '__main__':
    main()
