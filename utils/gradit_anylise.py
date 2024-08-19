import matplotlib.pyplot as plt
import numpy as np
import os, json, tqdm, re

def gradit_anylise():
    with open('/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/weight/COD/PNet/20%_with_ssloss/loss_records_0.json', 'r') as f:
        data1 = json.load(f)
    with open('/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/weight/COD/PNet/20%_with_ssloss/loss_records_1.json', 'r') as f:
        data2 = json.load(f)
    with open('/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/weight/COD/PNet/20%_with_ssloss/loss_records_2.json', 'r') as f:
        data3 = json.load(f)
    with open('/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/weight/COD/PNet/20%_with_ssloss/loss_records_3.json', 'r') as f:
        data4 = json.load(f)

    pair = {}
    keys = list(data1.keys())
    for key in tqdm.tqdm(keys):
        loss_sums = data1[key] + data2[key] + data3[key] + data4[key]
        flattened_data = {int(k): v for d in loss_sums for k, v in d.items()}
        sorted_data = dict(sorted(flattened_data.items()))
        pair[key] = list(sorted_data.values())
    
    with open('/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/weight/COD/PNet/20%_with_ssloss/loss_records.json', 'w') as f:
        json.dump(pair, f)
    return 1


def draw_grad_plot():
    with open('/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/weight/COD/PNet/20%_with_ssloss/loss_records.json', 'r') as f:
        data = json.load(f)
    with open('/mnt/jixie16t/dataset/COD/CAMO_COD_train/LabelNoiseTrainList/top_20_percent.txt', 'r') as f:
        clean_label = f.readlines()
        clean_label = [label.strip() for label in clean_label]
    with open('/mnt/jixie16t/dataset/COD/CAMO_COD_train/LabelNoiseTrainList/remaining_80_percent.txt', 'r') as f:
        noise_label = f.readlines()
        noise_label = [label.strip() for label in noise_label]
    clean_json = {}
    for key in clean_label:
        clean_json[key] = data[key]
    noise_json = {}
    for key in noise_label:
        noise_json[key] = data[key]
    clean_loss_values = list(clean_json.values())
    noise_loss_values = list(noise_json.values())
    clean_loss_values = np.array(clean_loss_values)
    clean_loss_values = clean_loss_values[:, :, 1]
    clean_loss_values = np.mean(clean_loss_values, axis=0)
    noise_loss_values = np.array(noise_loss_values)
    noise_loss_values = noise_loss_values[:, :, 1]
    noise_loss_values = np.mean(noise_loss_values, axis=0)
    # noise_loss_values = np.array(noise_loss_values)
    # noise_loss_values = np.mean(noise_loss_values, axis=0)
    # clean_loss_values = clean_loss_values
    # noise_loss_values = noise_loss_values
    clean_loss_values_grad = clean_loss_values[1:] - clean_loss_values[0:-1]
    clean_loss_values_grad = [0] + clean_loss_values_grad.tolist()
    noise_loss_values_grad = noise_loss_values[1:] - noise_loss_values[0:-1]
    noise_loss_values_grad = [0] + noise_loss_values_grad.tolist()
    plt.plot(clean_loss_values_grad[:50], label='clean', linestyle='dotted')
    plt.plot(noise_loss_values_grad[:50], label='noise', linestyle='dotted')
    
    # loss_values = list(data.values())
    # loss_values = np.array(loss_values)
    # loss_values = np.mean(loss_values, axis=0)
    # iou_improvements = [(loss_values[i] - loss_values[i - 1]) for i in range(1, len(loss_values))]
    # # loss_values_grad = loss_values[1:] - loss_values[0:-1]
    # # loss_values_grad = [0] + loss_values_grad.tolist()
    # # plt.plot(iou_improvements, label='all_label')
    plt.legend()
    plt.savefig('/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/weight/COD/PNet/20%_with_ssloss/loss_records.png')
    # return 1


def acc_plow():
    with open('/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/weight/COD/PNet/20%_with_ssloss/log.log', 'r') as file:
        logs = file.readlines()

    # 提取MAE值的正则表达式
    mae_pattern = re.compile(r'MAE: ([0-9]*\.?[0-9]+)')

    # 提取MAE值
    mae_values = []
    for line in logs:
        match = mae_pattern.search(line)
        if match:
            mae_values.append(float(match.group(1)))
    mae_values = np.array(mae_values)
    grad_mae_values = mae_values[1:] - mae_values[0:-1]
    grad_mae_values = [-i for i in grad_mae_values]
    # 绘制MAE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(grad_mae_values)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE vs. Epoch')
    plt.grid(True)
    plt.savefig('/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/weight/COD/PNet/20%_with_ssloss/grad_mae_values.jpg')


if __name__ == '__main__':
    # gradit_anylise()
    draw_grad_plot()
