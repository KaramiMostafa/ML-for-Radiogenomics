"""
libraries : Pyradiomics
"""
import os
import dicom2nifti
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from __future__ import print_function
import collections
import csv
import logging
import os
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import glob

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


    ''' This function generate a csv file with two columns: Image and Mask
        directory of different sequence type and different segmented brain 
        tumor will be added as a new row to the csv file.
    '''
    def generate_csv(self):
        with open(os.path.join(self.path_result,'radiomics_features_task1.csv'),'a',newline='') as csvfile:
            
        #     # creating the column heads
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
            
    ''' 
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
