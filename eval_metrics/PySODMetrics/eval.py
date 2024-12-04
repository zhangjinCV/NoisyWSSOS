import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure, IoU


# preds_root = ['/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/1%/CAMO',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/1%/CHAMELEON_TestingDataset',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/1%/COD10K',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/1%/NC4K',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/5%/CAMO',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/5%/CHAMELEON_TestingDataset',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/5%/COD10K',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/5%/NC4K',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/10%/CAMO',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/10%/CHAMELEON_TestingDataset',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/10%/COD10K',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/10%/NC4K',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/20%/CAMO',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/20%/CHAMELEON_TestingDataset',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/20%/COD10K',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/COD/PNet/20%/NC4K'
#               ]
# saves_path = ['/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/1%/CAMO/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/1%/CHAMELEON_TestingDataset/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/1%/COD10K/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/1%/NC4K/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/5%/CAMO/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/5%/CHAMELEON_TestingDataset/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/5%/COD10K/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/5%/NC4K/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/10%/CAMO/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/10%/CHAMELEON_TestingDataset/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/10%/COD10K/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/20%/NC4K/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/20%/CAMO/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/20%/CHAMELEON_TestingDataset/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/20%/COD10K/',
#               '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/PNet/20%/NC4K/',
#               ]
# masks_root = ['/mnt/jixie16t/dataset/COD/CAMO/mask',
#               '/mnt/jixie16t/dataset/COD/CHAMELEON_TestingDataset/mask',
#               '/mnt/jixie16t/dataset/COD/COD10K/mask',
#               '/mnt/jixie16t/dataset/COD/NC4K/mask'
#               ] * 4
preds_root = [
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/1%/CVC-300',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/1%/CVC-ClinicDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/1%/CVC-ColonDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/1%/ETIS-LaribPolypDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/1%/Kvasir',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/5%/CVC-300',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/5%/CVC-ClinicDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/5%/CVC-ColonDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/5%/ETIS-LaribPolypDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/5%/Kvasir',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/10%/CVC-300',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/10%/CVC-ClinicDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/10%/CVC-ColonDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/10%/ETIS-LaribPolypDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/10%/Kvasir',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/20%/CVC-300',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/20%/CVC-ClinicDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/20%/CVC-ColonDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/20%/ETIS-LaribPolypDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/20%/Kvasir',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/100%/CVC-300',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/100%/CVC-ClinicDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/100%/CVC-ColonDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/100%/ETIS-LaribPolypDB',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/Polyp/PNet/100%/Kvasir',
              ]
saves_path = ['/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/1%/CVC-300/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/1%/CVC-ClinicDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/1%/CVC-ColonDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/1%/ETIS-LaribPolypDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/1%/Kvasir/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/5%/CVC-300/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/5%/CVC-ClinicDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/5%/CVC-ColonDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/5%/ETIS-LaribPolypDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/5%/Kvasir/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/10%/CVC-300/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/10%/CVC-ClinicDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/10%/CVC-ColonDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/10%/ETIS-LaribPolypDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/10%/Kvasir/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/20%/CVC-300/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/20%/CVC-ClinicDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/20%/CVC-ColonDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/20%/ETIS-LaribPolypDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/20%/Kvasir/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/100%/CVC-300/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/100%/CVC-ClinicDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/100%/CVC-ColonDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/100%/ETIS-LaribPolypDB/',
              '/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/score_records/Polyp/PNet/100%/Kvasir/',
]
masks_root = [
    '/mnt/jixie16t/dataset/Polyp/TestDataset/CVC-300/mask',
    '/mnt/jixie16t/dataset/Polyp/TestDataset/CVC-ClinicDB/mask',
    '/mnt/jixie16t/dataset/Polyp/TestDataset/CVC-ColonDB/mask',
    '/mnt/jixie16t/dataset/Polyp/TestDataset/ETIS-LaribPolypDB/mask',
    '/mnt/jixie16t/dataset/Polyp/TestDataset/Kvasir/mask'
] * 5
for pred_root, mask_root, save_path in zip(preds_root, masks_root, saves_path):
    print("eval on dataset: {}".format(pred_root))
    mask_name_list = sorted(os.listdir(pred_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    I = IoU()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, 0)
        pred = cv2.imread(pred_path, 0)
        FM.step(pred=pred, gt=mask)
        # WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)
        I.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]
    iou = I.get_results()['iou']
    results =  "&  " + "%.3f" % mae + "  &  " + "%.3f" % iou + "  &  " + "%.3f" % fm["curve"].mean() +  "  &  " + "%.3f" % sm
    print(results)
    os.makedirs(save_path, exist_ok=True)
    file=open(os.path.join(save_path, "results.txt"), "w")
    file.write(pred_root + '\n')
    file.write(str(results)+'\n')
    file.close()

print("Eval finished!")