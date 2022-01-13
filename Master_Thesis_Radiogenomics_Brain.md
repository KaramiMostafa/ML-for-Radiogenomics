## <center><p align = "center"> Machine Learning Algorithms for Radiogenomics: <br />  Application to Prediction of the MGMT promoter methylation status in mpMRI scans </p> </center>

Proposal title of Master thesis:
- Application of Machine learning algorithms for radiogenomics


### :busts_in_silhouette: Team

Mostafa Karami, Liliana Caldeira

### :rocket: Purpose/Aim
* 
*

### :chart_with_upwards_trend: Data

**Folder**: 

### :wrench: Methods
Radiomics/Machine learning, Segmentation, Classification

#### Workflow
- [ ] 

----

**Issues**

* 


----
**WORK PLAN**

- [x] Read BRATS challenge https://www.med.upenn.edu/cbica/brats2021/

- [x] Download Task2 (130GB):

https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/code?competitionId=29653&sortBy=voteCount


--- 1 week ---

----


- [x] Share code through Github (username: LilianaCaldeira)
https://github.com/datascienceukk

- [x] LC - Find simple MRI introduction to read (contrasts, sequences)

Pydicom - Dicom headers:
https://pydicom.github.io/
What kind of information is there?

- [x] Convert Metadata -> Txt, Csv/Excel (only part that are general)

- [x] Convert dicom to nifti -> LC will upload draft script in Github


- [x] Python visualization (3D-> 
3 subplot views/directions of 2D):

- SimpleITK
https://simpleitk.readthedocs.io/en/master/link_DicomSeriesReadModifyWrite_docs.html

- Nibabbel

--- 2 week goal ---

- [x] Download Task1 for the segmentations ground truth (150GB):
https://www.synapse.org/#!Synapse:syn25829067/wiki/610863
(if not possible, LC will ask Pranali)
- [x] Separate visualization from dicom2nifti conversion
- [x] Read Docstrings and style guide and apply to code https://goodresearch.dev/setup.html
- [x] Correct Axial, Sagittal, Coronal in figures
- [x] Save figures for all patients in jpeg in the folders
- [x]  DFKZ - brain tumour segmentation:  
https://github.com/datascienceukk/HD-GLIO-AUTO (can be tried on googlecollab)
- [x] Pyradiomics install and document, preparing script.

--- 3 week goal ---

- [ ] visualization of the tumour segmentation:
    - [ ] find best tumour slice to visualize for each view (sag, trans, cor) 
    - [x] compare jpegs to Task 2
    - [x] find out more about the tissues segmented: core, edema, etc...
- [X] Getting info about all images (voxel size, image size)

--- 4 week goal ---

- [ ] Getting info about all images for Task1 (voxel size, image size)
- [ ] visualization of the tumour segmentation:
    - [ ] find best tumour slice to visualize for each view (sag, trans, cor) for tumour core:
            - create image with just core
            - sum over one direction to create a vector
            - do the sum for all 3 different directions
            - get position of the max in each direction
            - use those position values for visualization 
- [ ] Resample all images of Task 2 to common grid 1x1x1 mm3
        - Upload the resample python script (Liliana) 
        - Compare images from task 1 and task 2 with resampled version
- [ ] More info about regions here:
    https://www.med.upenn.edu/cbica/brats2021/
    - [ ] create segmentation WholeTumour (1+2+4)
    - [ ] create segmentation tumour core (1+2)


---- 5 week goal --- 

- [ ] LC letter of support Mostafa Karami


https://pyradiomics.readthedocs.io/en/latest/
To extract features from images (Nifti)


Preprocessing of features:

https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

Simple approach to start:

https://cancerimagingjournal.biomedcentral.com/articles/10.1186/s40644-020-00374-3

Paper "Best" approach: 

https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-8/issue-03/031905/Improving-performance-and-generalizability-in-radiogenomics--a-pilot-study/10.1117/1.JMI.8.3.031905.full
 

