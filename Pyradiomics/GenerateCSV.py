import os
import csv
import fnmatch
path = r'U:\Lung_missing'
with open(r'U:\Lung_missing\Radiomics_LungTest.csv','a',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image','Mask'])
    for dirpath, dirs, filenames in os.walk(path, topdown=True):
        dirs.sort()
        pattern = "*COPD_*"
        for entry in filenames:
            if fnmatch.fnmatch(entry, pattern):
                mask=dirpath +'\\'+ entry
                image=dirpath +'\\'+ entry.split("_")[0]+".nii.gz"
                print(image, mask)                
                writer.writerow([image] + [mask])