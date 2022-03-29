import matplotlib.pyplot as plt
import yaml
import os



workspace = "/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion"
num_samples = 192
log = os.path.join(workspace, 'log/imagenet1000_classifier256x256_channel128_upperbound/predict/model500000_imagenet1000_stepsddim25_sample{}_selectedClass'.format(num_samples))
legends = []

plt.figure()
for class_id in range(3):
    fid = []
    for scale in range(1,21):
        result_name = 'result_scale{}.0_class{}_stepsddim25_sample{}.yaml'.format(scale, class_id, num_samples)
        result_path = os.path.join(log,result_name)
        with open(result_path, "r") as stream:
            try:
                result_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        fid.append(result_dict['fid'])
        print(result_dict)
    plt.plot(fid)
    plt.xlabel('classifier scale')
    plt.ylabel(fid)
    legends.append('sample{}_class{}'.format(num_samples, class_id))

plt.legend(legends)

plt.show()







