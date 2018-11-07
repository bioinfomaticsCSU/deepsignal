# -*- coding: utf-8 -*-
# @Author: huangneng
# @Date:   2018-09-25 08:40:34
# @Last Modified by:   hn
# @Last Modified time: 2018-09-26 11:42:53

import os

model_path = "/home/huangneng/master_deepsignal/deepsignal/last_v_models/ecoli2human_17mer"
result_path = "/home/huangneng/master_deepsignal/deepsignal/last_v_results/ecoli_hx1_single_1"
test_dir = "/home/huangneng/data/human_hx1.test.5xs/BJXWZ.C789101112D1E13"
model_name = '4'

for subdir in os.listdir(test_dir):
    if not subdir.endswith(".scgs"):
        continue
    test_path = os.path.join(test_dir, subdir)
    out_name = os.path.join(result_path, '.'.join(subdir.split('.'))+'.result')
    # print(test_path)
    # print(out_name)
    cmd = "CUDA_VISIBLE_DEVICES=1 python ../predict.py -i {input}/ -o {model}/ -r {output} -n {modelname} -x 17 -y 375 -z 60 ".format(
        input=test_path, model=model_path, modelname=model_name, output=out_name)
    print(cmd)
    os.system(cmd)
