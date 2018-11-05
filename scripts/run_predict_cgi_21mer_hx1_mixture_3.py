# -*- coding: utf-8 -*-
'''
/*
 * @Author: huangneng 
 * @Date: 2018-10-26 15:27:23 
 * @Last Modified by: huangneng
 * @Last Modified time: 2018-10-28 11:22:26
 */
'''

import os

model_path = "/home/huangneng/master_deepsignal/deepsignal/last_v_models/cgi_21mer_mixture"
result_path = "/home/huangneng/master_deepsignal/deepsignal/last_v_results/cgi_21mer_hx1_mixture_3"
test_dir = "/home/huangneng/data/human_hx1.test.bn21.5xs/BJXWZ.cgi.3.22B408D91009E2"
model_name = '19'

for subdir in os.listdir(test_dir):
    if not subdir.endswith(".mcgs"):
        continue
    test_path = os.path.join(test_dir, subdir)
    out_name = os.path.join(result_path, '.'.join(subdir.split('.'))+'.result')
    # print(test_path)
    # print(out_name)
    cmd = "CUDA_VISIBLE_DEVICES=2 python ../predict.py -i {input}/ -o {model}/ -r {output} -n {modelname} -x 21 -y 512 -z 60 ".format(
        input=test_path, model=model_path, modelname=model_name, output=out_name)
    print(cmd)
    os.system(cmd)
