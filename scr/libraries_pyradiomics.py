"""
libraries : Pyradiomics
"""
import os
# import dicom2nifti
# import pandas as pd
# import matplotlib.pyplot as plt
# import nibabel as nib
# from __future__ import print_function
import collections
import csv
import logging
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import glob
import numpy as np

""" path_result : storing the result in a specific directory
    seg_path : different segmented images
    patientID : list of patiend's ID - Task 1 or 2
    train_task1 : NIFTI images for task 1 dataset
    sequence_types : sequence type of mMRI pictures
"""


class Pyradiomics:
    
    def __init__(self,path_result,seg_path,train_task1,sequence_types,patientID):
    
        self.path_result = r'D:\ICT\Thesis\Github\repo\Pyradiomics'
        self.seg_path = r'D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021'
        self.train_task1 = (r'D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021')
        self.sequence_types = ["flair", "t1", "t1ce", "t2"]
        self.patientID = os.listdir(
                    r'D:\ICT\Thesis\Data\Task1\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021')


    ''' <<generate_csv>> function generate a csv file with two columns: Image and Mask
        directory of different sequence type and different segmented brain 
        tumor will be added as a new row to the csv file.
        
        outcome:
            create a csv file with original image and different masks.
    '''
    def generate_csv(self):
        with open(os.path.join(self.path_result,'radiomics_features_task1.csv'),'a',newline='') as csvfile:
            
            # creating the column heads
            writer = csv.writer(csvfile)
            writer.writerow(['Image','Mask'])
            
            # filling each cell with the path of image and mask
            try:
                for patient in self.patientID:
                    for types in self.sequence_types:
                        
                        
                        img = os.path.join(self.train_task1, patient + '\\'
                                           + patient +'_'+ types + '.nii.gz')
                        
                        
                        mask = os.path.join(self.train_task1, patient + '\\' + 
                                             patient + '_seg.nii.gz')
                        mask_ED = os.path.join(self.train_task1, patient + '\\' + 
                                             patient + '_seg_ED.nii.gz')
                        mask_ET = os.path.join(self.train_task1, patient + '\\' + 
                                             patient + '_seg_ET.nii.gz')
                        mask_NCR = os.path.join(self.train_task1, patient + '\\' + 
                                             patient + '_seg_NCR.nii.gz')
                        
                        writer.writerow([img] + [mask])
                        writer.writerow([img] + [mask_ED])
                        writer.writerow([img] + [mask_ET])
                        writer.writerow([img] + [mask_NCR])
            except:
                pass
            
            
    """ Separating the segmented images of task 1 into four parts:
        1. image with the whole tumor (labes : 1+2+4)
        2. image with necrotic (NCR) parts of the tumor (labes : 1)
        3. image with peritumoral edematous/invaded tissue (ED) (labes : 2)
        4. image with enhancing tumor (ET) (labes : 4)
        5. image with tumor core (COR) (labels : 1+4)
    """
    def separate_seg(self):
        #%% loop through the input path folder to separate different area
        #   and relable it to 1 for feature extraction

        for patient in self.patientID:

            # reading the original image
            img = sitk.ReadImage(
                os.path.join(
                    self.train_task1,
                    patient + "/" + patient + "_seg.nii.gz",
                )
            )

            # getting the image data
            img_data = sitk.GetArrayFromImage(img)

            # relabel the whole tumor to 1
            img_whole_data = np.where((img_data != 0), 1, img_data)

            # keeping the nerotic part of the tumor and removing the rest
            img_NCR_data = np.where((img_data != 1), 0, img_data)

            # extracting the edema (label 2) and relabel it to 1
            img_ED_data = np.where(
                (np.where((img_data != 2), 0, img_data) == 2),
                1,
                np.where((img_data != 2), 0, img_data),
            )
            
            # extracting the enhacing part of the tumor (label 4) and relabel it to 1
            img_ET_data = np.where(
                (np.where((img_data != 4), 0, img_data) == 4),
                1,
                np.where((img_data != 4), 0, img_data),
            )

            # extracting the core of the tumor (label 1 & 4) and relabel it to 1
            img_COR_data = (
                np.where((img_data == 2), 0, img_data)
                & np.where((img_data == 1), 1, img_data)
                & np.where((img_data == 4), 1, img_data)
            )
            
            # getting the metadata of the original image and assign it to 
            # new segmented area as NIFTI file
            img_whole = sitk.GetImageFromArray(img_whole_data)
            img_whole.CopyInformation(img)
            img_NCR = sitk.GetImageFromArray(img_NCR_data)
            img_NCR.CopyInformation(img)
            img_ED = sitk.GetImageFromArray(img_ED_data)
            img_ED.CopyInformation(img)
            img_ET = sitk.GetImageFromArray(img_ET_data)
            img_ET.CopyInformation(img)
            img_COR = sitk.GetImageFromArray(img_COR_data)
            img_COR.CopyInformation(img)


            # saving all the nifti files in output path
            sitk.WriteImage(
                img_whole, os.path.join(self.train_task1, patient + "/" + patient + "_seg_whole.nii.gz")
            )

            sitk.WriteImage(
                img_NCR, os.path.join(self.train_task1, patient + "/" + patient + "_seg_NCR.nii.gz")
            )

            sitk.WriteImage(
                img_ED, os.path.join(self.train_task1, patient + "/" + patient + "_seg_ED.nii.gz")
            )

            sitk.WriteImage(
                img_ET, os.path.join(self.train_task1, patient + "/" + patient + "_seg_ET.nii.gz")
            )

            sitk.WriteImage(
                img_COR, os.path.join(self.train_task1, patient + "/" + patient + "_seg_COR.nii.gz")
            )   
            
            
            
    ''' <<feature_extraction>> function uses the csv file which contains the 
        directory to segmented images and differemt masks in order to extract
        radiomics features of each nifti image.
        
        outcome:
            create a csv file with all the features related to origanl images (and
            its filters)
    '''
    
    def feature_extraction():

        os.chdir(r"D:\ICT\Thesis\Github\repo\Pyradiomics")
        outPath = r"D:\ICT\Thesis\Github\repo\Pyradiomics"

        filescsv = glob.glob("radiomics_features_task1.csv")

        # filescsv=glob.glob('Radiomics_*.csv')
        for inFile in filescsv[:]:
            inputCSV = os.path.join(outPath, inFile)
            outputFilepath = os.path.join(outPath, "Results_" + inFile)
            progress_filename = os.path.join(outPath, "pyrad_log.txt")
            params = os.path.join(outPath, "exampleSettings", "Params.yaml")

            # Configure logging
            rLogger = logging.getLogger("radiomics")

            # Set logging level
            # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

            # Create handler for writing to log file
            handler = logging.FileHandler(filename=progress_filename, mode="w")
            handler.setFormatter(
                logging.Formatter("%(levelname)s:%(name)s: %(message)s")
            )
            rLogger.addHandler(handler)

            # Initialize logging for batch log messages
            logger = rLogger.getChild("batch")

            # Set verbosity level for output to stderr (default level = WARNING)
            radiomics.setVerbosity(logging.INFO)

            logger.info("pyradiomics version: %s", radiomics.__version__)
            logger.info("Loading CSV")

            flists = []
            try:
                with open(inputCSV, "r") as inFile:
                    cr = csv.DictReader(inFile, lineterminator="\n")
                    flists = [row for row in cr]
            except Exception:
                logger.error("CSV READ FAILED", exc_info=True)

            logger.info("Loading Done")
            logger.info("Patients: %d", len(flists))

            if os.path.isfile(params):
                extractor = featureextractor.RadiomicsFeatureExtractor(params)
            else:  # Parameter file not found, use hardcoded settings instead
                settings = {}
                # settings['binWidth'] = 25
                # settings['resampledPixelSpacing'] = [0.75, 0.75, 1]  # [3,3,3]
                # settings['interpolator'] = sitk.sitkBSpline
                # settings['correctMask'] = True
                settings["geometryTolerance"] = 1
                # settings['enableCExtensions'] = True

                extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
                # extractor.enableInputImages(wavelet= {'level': 2})

            #  logger.info('Enabled input images types: %s', extractor.enabledImageTypes)
            #  logger.info('Enabled features: %s', extractor.enabledFeatures)
            #  logger.info('Current settings: %s', extractor.settings)

            headers = None

            for idx, entry in enumerate(flists, start=1):

                logger.info(
                    "(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                    idx,
                    len(flists),
                    entry["Image"],
                    entry["Mask"],
                )

                imageFilepath = entry["Image"]
                maskFilepath = entry["Mask"]
                label = entry.get("Label", None)

                if str(label).isdigit():
                    label = int(label)
                else:
                    label = None

                if (imageFilepath is not None) and (maskFilepath is not None):
                    featureVector = collections.OrderedDict(entry)
                    featureVector["Image"] = os.path.basename(imageFilepath)
                    featureVector["Mask"] = os.path.basename(maskFilepath)

                    try:
                        featureVector.update(
                            extractor.execute(imageFilepath, maskFilepath, label)
                        )

                        with open(outputFilepath, "a") as outputFile:
                            writer = csv.writer(outputFile, lineterminator="\n")
                            if headers is None:
                                headers = list(featureVector.keys())
                                writer.writerow(headers)

                            row = []
                            for h in headers:
                                row.append(featureVector.get(h, "N/A"))
                            writer.writerow(row)
                    except Exception:
                        logger.error("FEATURE EXTRACTION FAILED", exc_info=True)
                        
                        
