from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR
import shutil
import os
import json
import zipfile

def create_zip_archive(zip_filename, files_to_zip = ['answer.json', 'answer_detail.json',"recall.txt"], folder_to_zip = 'result'):

    # 创建并打开zip文件
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 添加文件到zip文件
        for file in files_to_zip:
            file_path = os.path.abspath(file)
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))

        # 添加文件夹到zip文件
        folder_path = os.path.abspath(folder_to_zip)
        for foldername, subfolders, filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

    print(f'Zip文件已创建并保存到: {zip_filename}')

if __name__ == "__main__":
    checkpoints_dir = "/root/share/Project/OCR_Project/bttr_ocr/BTTR_STR/lightning_logs/version_2_allaug/checkpoints"
    ckp_paths = []
    lmdb_test_path = "/root/share/Project/OCR_Project/bttr_ocr/BTTR_STR/data/for_eval"

    for file in os.listdir(checkpoints_dir):
        ckp_paths.append(os.path.join(checkpoints_dir, file))
    for i, ckp_path in enumerate(ckp_paths):
        trainer = Trainer(logger=False, gpus=1)

        os.system(f"rm -rf result")
        os.system(f"mkdir result")
        dm = CROHMEDatamodule(datapath=lmdb_test_path)
        model = LitBTTR.load_from_checkpoint(ckp_path)
        trainer.test(model, datamodule=dm)
        savename = os.path.basename(file) + "_" + os.path.basename(ckp_path)[:-5] + ".zip" 
        create_zip_archive(savename)
        