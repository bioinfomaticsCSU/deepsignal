import os

model_path = "/home/huangneng/master_deepsignal/deepsignal/last_v_models/hx1_bn360_mad_acg"
result_path = "/home/huangneng/master_deepsignal/deepsignal/last_v_results/acg_na12878_single_3"
test_dir = "/home/huangneng/data/na12878.norm_mad/na12878.10x.3"
model_name = '5'

for subdir in os.listdir(test_dir):
    if not subdir.endswith(".scgs"):
        continue
    test_path = os.path.join(test_dir, subdir)
    out_name = os.path.join(result_path, '.'.join(subdir.split('.'))+'.result')
    # print(test_path)
    # print(out_name)
    cmd = "CUDA_VISIBLE_DEVICES=2 python ../predict.py -i {input}/ -o {model}/ -r {output} -n {modelname} -x 17 -y 360 -z 60 ".format(
        input=test_path, model=model_path, modelname=model_name, output=out_name)
    print(cmd)
    os.system(cmd)
