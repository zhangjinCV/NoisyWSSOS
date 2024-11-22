import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


methods=["ANet"]
for method in methods: 
    for _data_name in ['pseudo_label_80%']:
        print("eval-dataset: {}".format(_data_name))
        mask_root = '/mnt/jixie16t/dataset/DIS5K/DIS-TR/gt/'
        pred_root = "/mnt/jixie16t/zj/zj/works_in_phd/NoisyCOS/results/DIS5K/ANet/coreset/remaining80/mask/"
        mask_name_list = sorted(os.listdir(pred_root))
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
        # file=open("/mnt/550aa7b7-3fbe-43b7-86bd-198efb9b4305/zj/works_in_phd/Hacker/SINet/SINet-V2-main/SINet-V2-main/scores/eval_results.txt", "a")
        # file.write(method+' '+_data_name+' '+str(results)+'\n')

print("Eval finished!")