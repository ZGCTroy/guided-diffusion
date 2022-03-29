import matplotlib.pyplot as plt
import yaml
import os

log_root = "/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier128x128_channel128_baseline/predict/find_optimal_entropy_scale"

num_samples = 5000
iteration = 300000

fig = plt.figure(figsize=(40, 20))
ax = plt.subplot(2,1,1)
ax.set_title('stepsddim25, conditional')
steps = 'ddim25'
fid = []
scales = [0.5, 0.75, 1.0, 1.1, 1.2, 1.25, 1.5, 1.75]
for idx, scale in enumerate(scales):
    predict_name = "conditional_model{}_steps{}_scale{}_sample{}_entropyScale".format(iteration, steps, scale,num_samples)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*')
plt.xlabel('entropy scale')
plt.ylabel('fid')


ax = plt.subplot(2,1,2)
ax.set_title('steps250, conditional')
steps = '250'
fid = []
scales = [0.3, 0.4, 0.5]
for idx, scale in enumerate(scales):
    predict_name = "conditional_model{}_steps{}_scale{}_sample{}_entropyScale".format(iteration, steps, scale,num_samples)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*')
plt.xlabel('entropy scale')
plt.ylabel('fid')

plt.savefig('/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/optimal_entropy_scale_128x128.pdf')
plt.show()
