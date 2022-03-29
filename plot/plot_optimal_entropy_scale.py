import matplotlib.pyplot as plt
import yaml
import os

log_root = "/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_upperbound/predict/find_optimal_entropy_scale"

num_samples = 5000
iteration = 500000
import seaborn as sns

sns.set_theme(style="white")
fig = plt.figure(figsize=(45, 25), dpi=600)
fontsize = 30
plt.rcParams['font.size'] = fontsize

ax = plt.subplot(2, 4, 1)
steps = 'ddim25'
ax.set_title('EDS, unconditional stepsddim25', fontsize=fontsize)
fid = []
scales = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
for scale in scales:
    predict_name = "model{}_imagenet1000_steps{}_sample{}_scale{}_entropyScale".format(iteration, steps, num_samples, scale)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*', lw=3, )
# plt.xlabel('entropy scale', fontsize=fontsize)
plt.ylabel('fid', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# plt.show()

# plt.figure()
ax = plt.subplot(2, 4, 2)
steps = '250'
ax.set_title('EDS, unconditional steps250', fontsize=fontsize)
fid = []
scales = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
for scale in scales:
    predict_name = "model{}_imagenet1000_steps{}_sample{}_scale{}_entropyScale".format(iteration, steps, num_samples, scale)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*', lw=3, )
# plt.xlabel('entropy scale', fontsize=fontsize)
# plt.ylabel('fid', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# plt.show()

# plt.figure()
ax = plt.subplot(2, 4, 3)
steps = 'ddim25'
ax.set_title('EDS, conditional stepsddim25', fontsize=fontsize)
fid = []
scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
for scale in scales:
    predict_name = "conditional_model{}_imagenet1000_steps{}_sample{}_scale{}_entropyScale".format(iteration, steps, num_samples, scale)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*', lw=3, )
# plt.xlabel('entropy scale', fontsize=fontsize)
# plt.ylabel('fid', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# plt.show()

# plt.figure()
ax = plt.subplot(2, 4, 4)
steps = '250'
ax.set_title('EDS, conditional steps250', fontsize=fontsize)
fid = []
scales = [0.25, 0.5, 0.75, 1.0]
for scale in scales:
    predict_name = "conditional_model{}_imagenet1000_steps{}_sample{}_scale{}_entropyScale".format(iteration, steps, num_samples, scale)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*', lw=3, )
# plt.xlabel('entropy scale', fontsize=fontsize)
# plt.ylabel('fid', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# plt.show()

##################

log_root = "/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_entropyConstraintTrain0.2/predict/find_optimal_entropy_scale"

num_samples = 5000
iteration = 500000

# plt.figure()
ax = plt.subplot(2, 4, 5)
steps = 'ddim25'
ax.set_title('ECT + EDS, unconditional stepsddim25', fontsize=fontsize)
fid = []
scales = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
for scale in scales:
    predict_name = "model{}_imagenet1000_steps{}_sample{}_scale{}_entropyScale".format(iteration, steps, num_samples, scale)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*', lw=3, )
plt.xlabel('entropy scale', fontsize=fontsize)
plt.ylabel('fid', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# plt.show()


# plt.figure()
ax = plt.subplot(2, 4, 6)
steps = 250
ax.set_title('ECT + EDS, unconditional steps250', fontsize=fontsize)
fid = []
scales = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
for scale in scales:
    predict_name = "model{}_imagenet1000_steps{}_sample{}_scale{}_entropyScale".format(iteration, steps, num_samples, scale)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*', lw=3, )
plt.xlabel('entropy scale', fontsize=fontsize)
# plt.ylabel('fid', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# plt.show()

# plt.figure()
ax = plt.subplot(2, 4, 7)
steps = 'ddim25'
ax.set_title('ECT + EDS, conditional stepsddim25', fontsize=fontsize)
fid = []
scales = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]

for scale in scales:
    predict_name = "conditional_model{}_imagenet1000_steps{}_sample{}_scale{}_entropyScale".format(iteration, steps, num_samples, scale)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*', lw=3, )
plt.xlabel('entropy scale', fontsize=fontsize)
# plt.ylabel('fid', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
# plt.show()


# plt.figure()
ax = plt.subplot(2, 4, 8)
steps = 250
ax.set_title('ECT + EDS, conditional steps250', fontsize=fontsize)
fid = []
scales = [0.25, 0.5, 0.75, 0.8, 0.9, 1.0, 1.25]

for scale in scales:
    predict_name = "conditional_model{}_imagenet1000_steps{}_sample{}_scale{}_entropyScale".format(iteration, steps, num_samples, scale)
    result_name = 'result_scale{}_steps{}_sample{}.yaml'.format(scale, steps, num_samples)
    result_path = os.path.join(log_root, predict_name, result_name)
    with open(result_path, "r") as stream:
        try:
            result_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    fid.append(result_dict['fid'])
    print(result_dict)

plt.plot(scales, fid, marker='*', lw=3, )
plt.xlabel('entropy scale', fontsize=fontsize)
# plt.ylabel('fid', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.savefig(
    '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/imgs/optimal_entropy_scale.pdf',
    bbox_inches='tight'
)
# plt.show()
