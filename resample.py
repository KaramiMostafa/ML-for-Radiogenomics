# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:50:34 2019

@author: xuh
"""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def map_origin(itk_image, out_spacing, out_size):
    """adjust the origin for the output image 
    
    Adjust the origin for the output image using the dimension and spacing 
    of the output image, as well as the direction vector. The method is done
    by aligning the center of image before and after resampling, because they
    should be the same point in the real space. Note that the origin of the image
    is the center point's coordinate of the left bottom voxel for 2D image. For
    3D, Z axis is also at the boarder of the image. 
    
    Args:
        itk_image: the image object in SimpleITK, which we can fetch original
            image dimension, direction vector, etc. 
        out_spacing: the spacing for the output image
        out_size: the size of the output image, if it is the default None, then the image size is
            determined by the spacing before and after resmapling, as well as the original image
            size    
    """
    or_spacing = np.asarray(itk_image.GetSpacing())
    or_size = np.asarray(itk_image.GetSize())
    or_origin = np.asarray(itk_image.GetOrigin())
    or_direction = np.asarray(itk_image.GetDirection())
    or_direction_x = or_direction[0:3]
    or_direction_y = or_direction[3:6]
    or_direction_z = or_direction[6:9]    
    
    ot_spacing = np.asarray(out_spacing)
    ot_size = np.asarray( out_size )

    # aligning the center
    ot_origin = (or_origin + ((or_size[0]*or_spacing[0])*or_direction_x +
                 (or_size[1]*or_spacing[1])*or_direction_y +
                 (or_size[2]*or_spacing[2])*or_direction_z)/2. -
                 ((ot_size[0]*ot_spacing[0])*or_direction_x +
                  (ot_size[1]*ot_spacing[1])*or_direction_y +
                  (ot_size[2]*ot_spacing[2])*or_direction_z)/2.)
    return ot_origin
    
    
    


def resample_image(itk_image, out_spacing=None, out_size = None, 
                   interpolator = sitk.sitkNearestNeighbor ):
    """resample the 3D image into another one
    
    Resample the 3D image into another 3D image with different spacing, image size using simpleITK
    
    Args:
        itk_image: the image object in SimpleITK
        out_spacing: the spacing for the output image
        out_size: the size of the output image, if it is the default None, then the image size is
            determined by the spacing before and after resmapling, as well as the original image
            size
        iterpolator: the supported interpolator by SimpleITK is listed below:
                sitk.sitkNearestNeighbor
                sitk.sitkLinear
                sitk.sitkBSpline
                sitk.sitkGaussian
                sitk.sitkHammingWindowedSinc
                sitk.sitkBlackmanWindowedSinc
                sitk.sitkCosineWindowedSinc
                sitk.sitkWelchWindowedSinc
                sitk.sitkLanczosWindowedSinc
            The performance of different interpolator is given by 
                https://simpleitk-prototype.readthedocs.io/en/latest/user_guide/transforms/plot_interpolation.html
    
    """
    # return the image, if we do not need to adjust the spacing or the dimension
    if out_spacing is None and out_size is None:
        return itk_image
        
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
#    original_origin = itk_image.GetOrigin()
    out_origin = itk_image.GetOrigin()
    if out_spacing is None:
        out_spacing = original_spacing
    
    if out_size is None:
        out_size = [int(np.round(original_size[0]*(original_spacing[0]/out_spacing[0]))),
                    int(np.round(original_size[1]*(original_spacing[1]/out_spacing[1]))),
                    int(np.round(original_size[2]*(original_spacing[2]/out_spacing[2])))]
    else:
        # udpate the center
        out_origin = tuple( map_origin(itk_image, out_spacing, out_size))
            
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(interpolator)
    
    return resample.Execute(itk_image) 
        

                         

if __name__=="__main__":
    
    #******************************************************************
    # input file
    file_path = 'T1.nii.gz'
    
    # write the DICOM image out
    out_file = 'T1_resampled.nii.gz'    
    
    # spacing configuration
#    out_spacing = None
    out_spacing = (2, 2, 2)
#    out_spacing = (0.5, 0.5, 0.5)
#    out_spacing = (1.5, 1.2, 1.2)    
   
    
    # size configuration
    out_size = None # put None, the program will decide the size of output
#    out_size = (512, 512, 350)
#    out_size = (400, 400, 400)
    
    # interpolator configuration
    interpolator = sitk.sitkBSpline
#    interpolator = sitk.sitkNearestNeighbor
    #******************************************************************
    
    # SimpleITK reader
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")    
    reader.SetFileName(file_path)
    raw_img = reader.Execute();
    
    print('*** Information of raw image ***')
    print('Image size is ', raw_img.GetSize() )
    print('Image spacing is ', raw_img.GetSpacing() )  
    print('Image origin is ', raw_img.GetOrigin() )
    print('Image direction is ', raw_img.GetDirection() )
    
    plt.figure()
    nda1 = sitk.GetArrayFromImage(raw_img)
    plt.imshow(nda1[:,:, int(nda1.shape[2]/2) ])    
    plt.show()


    
    print('Resampling the image')
    rsample_img = resample_image(raw_img, out_spacing=out_spacing, out_size=out_size,
                                 interpolator=interpolator)

    print('\n*** Information of resampled image ***')    
    print('Image size is ', rsample_img.GetSize() )
    print('Image spacing is ', rsample_img.GetSpacing() )      
    print('Image origin is ', rsample_img.GetOrigin() )
    print('Image direction is ', rsample_img.GetDirection() )    
        
    plt.figure()
    nda2 = sitk.GetArrayFromImage(rsample_img)
    plt.imshow(nda2[:,:, int(nda2.shape[2]/2) ])    
    plt.show()

    

    print("*** Writing the resampled image")
    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_file)
    writer.Execute(rsample_img)
    print("*** Writing finished\n")  
    
    print("*** Updating the header")
    ud_img = nib.load(out_file)
    ud_img.set_qform( ud_img.get_qform(), code='aligned' )
    ud_img.set_sform( ud_img.get_qform(), code='scanner' )
    nib.save( ud_img, out_file)
    print(ud_img.header)    
    
#    # SimpleITK reader
#    reader = sitk.ImageFileReader()
#    reader.SetImageIO("NiftiImageIO")    
#    reader.SetFileName(out_file)
#    read_back_img = reader.Execute();    
#    
#    plt.figure()
#    nda3 = sitk.GetArrayFromImage(read_back_img)
#    plt.imshow(nda3[:,:, int(nda3.shape[2]/2) ])    
#    plt.show()   
        