import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


methods=["1%", "5%", "10%", "20%", '50%']
for method in methods: 
    for _data_name in ['CAMO_TestingDataset','CHAMELEON_TestingDataset','COD10K_Test','NC4K']:
        print("eval-dataset: {}".format(_data_name))
        mask_root = '/home/zj/data/COD/{}/{}/'.format(_data_name,"mask") # change path
        pred_root = '/home/zj/data/COD/res/step2/fix_q/{}/{}/'.format(method, _data_name) # change path
        mask_name_list = sorted(os.listdir(mask_root))
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=mask)
            # WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            M.step(pred=pred, gt=mask)

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]

        results =  "&  " + "%.3f" % mae + "  &  " + "%.3f" % em["curve"].mean() + "  &  " + "%.3f" % fm["curve"].mean() +  "  &  " + "%.3f" % sm
        

        print(results)
        file=open("/mnt/550aa7b7-3fbe-43b7-86bd-198efb9b4305/zj/works_in_phd/ECCV2024/scores/eval_results.txt", "a")
        file.write(method+' '+_data_name+' '+str(results)+'\n')

print("Eval finished!")