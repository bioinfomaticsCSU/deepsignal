# -*- coding: utf-8 -*-
'''
/*
 * @Author: huangneng 
 * @Date: 2018-10-26 15:27:23 
 * @Last Modified by: huangneng
 * @Last Modified time: 2018-10-29 10:10:44
 */
'''

import os

model_path = "/home/huangneng/master_deepsignal/deepsignal/last_v_models/whole_genome_17mer_mixture"
result_path = "/home/huangneng/master_deepsignal/deepsignal/last_v_results/whole_20th_chrom_mixture"
test_dir = "/home/huangneng/data/na12878.test.30x/na12878.total/hx1.cpg.tem.test.bn17.sn375.total.chr20.mcgs"
model_name = '2'

test_path = '/home/huangneng/data/na12878.test.30x/na12878.total/hx1.cpg.tem.test.bn17.sn375.total.chr20.mcgs'
out_name = result_path+'/result.txt'
# print(test_path)
# print(out_name)
cmd = "CUDA_VISIBLE_DEVICES=2 python ../predict.py -i {input}/ -o {model}/ -r {output} -n {modelname} -x 17 -y 375 -z 60 ".format(
    input=test_path, model=model_path, output=out_name, modelname=model_name)
print(cmd)
os.system(cmd)
