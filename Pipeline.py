################################################################################
# Author:  Joy Roy, William Reynolds, Rafael Ceschin
# Purpose: The following is a pipeline made by PIRC at UPMC CHP in order to
#          preprocess BOLD fMRIs and to use those outputs for similarity 
#          matrix generation.
#
# This pipeline is currently being editted to more closely fit Power et al 2012
#
# Contact: jor115@pitt.edu
# Acknowledgments: Ansh Patel from The Hillman Academy contributed to this work.
################################################################################

################################################################################
##TO DO: 
# 1. Allow configuration of data/output/workingdir location
# 2. Allow configuration of template/segmentation files
# 3. Allow configuration of verbosity
# 4. Allow configuration of intermediate product storage
# 5. Take a config file to change paramters
# 6. (Optional) Take inline arguments from terminal
# 7. (Optional) Print blurb at beginning and end of the pipeline things users
#     should be aware of
##
################################################################################

import nibabel as nib
import nipy as nipy
import nipype.algorithms.rapidart as rpd    # rapidart module
import nipype.interfaces.io as nio          # Data i/o
import nipype.interfaces.fsl as fsl         # fsl
import nipype.interfaces.utility as util    # utility
import nipype.pipeline.engine as pe         # pypeline engine
import numpy as np
import os, sys


# ******************************************************************************
# ARGUMENTS AND CONFIGURATIONS
# ******************************************************************************

#sets the directories that will be used in the pipeline 
home_dir = os.getcwd()
data_dir = '/data' #the directory where the data is located
template_path = '/app/Template/MNI152lin_T1_2mm_brain.nii.gz' #the path where the template is located
segment_path = '/app/Template/AAL3v1_CombinedThalami_444.nii.gz'#template where thalami regions combined #added by joy

#############TEMPORARY FOR TESTING####################
data_dir = '/Volumes/Elements/FunctionalConnectome/preprocessing/data/finalDir/sample_datset'
template_path = '/Volumes/Elements/FunctionalConnectome/docker/py3_docker-master/Template/MNI152lin_T1_2mm_brain.nii.gz'
segment_path = '/Volumes/Elements/FunctionalConnectome/docker/py3_docker-master/Template/AAL3v1_CombinedThalami_444.nii.gz'
#############TEMPORARY FOR TESTING####################
# The pipeline graph and workflow directory will be outputted here
os.chdir(home_dir) #sets the directory of the workspace to the location of the data
# Sets default output to a compressed NIFTI
fsl.FSLCommand.set_default_output_type('NIFTI_GZ') #sets the default output type to .nii.gz
# NOTE: These lines control where the output is sorted.
# The derivatives folder should be a directory outside of the subjects directory
# This helps allows us to keep inputs and outputs in seperate locations.
# It is named 'derivatives' in accordance to BIDS (although this pipeline does
# not assume BIDS input YET)
derivatives_dir = os.path.join(data_dir, 'derivatives')
# Leave this blank if you do not want an extra directory of outputs
# We suggest you keep it incase youre running mulitple pipelines 
# together on the same input files. This will differentiate pipeline outputs
# Suggested names: 'FunkyConnect' or 'FunkyBrainSpawn' or 'JoyIsCool'
OUTFOLDERNAME = 'output'
# NOTE: THIS ALLOWS FOR ALL INTERMEDIATE OUTPUTS TO BE SAVED
SAVE_INTERMEDIATES = True
# NOTE: This is necessary to keep track of the original template range
MAX_SEGMENT_VAL = nib.load(segment_path).get_fdata().max()
# NOTE: The following looks for Niftis in the data dir 
# This does not include any niftis in the output directory
subject_list_abs = []
for dirpath, dirnames, filenames in os.walk(data_dir):
    for filename in [f for f in filenames if '.nii' in f]:
        if derivatives_dir in dirpath:
            continue
        else:
            filepath = os.path.join(dirpath, filename)
            subject_list_abs.append(filepath)
subject_list_abs = sorted(subject_list_abs)
# print(subject_list_abs)

# ******************************************************************************
# HELPER FUNCTIONS
# ******************************************************************************

#Note: outputs the home directory of the data to output all results to
def GenerateOutDir(base_outputdir, image_path):
    import os
    subj = image_path.split('/')[-3]
    sess = image_path.split('/')[-2]
    out_dir = os.path.join(base_outputdir, subj, sess)
    os.makedirs(out_dir, exist_ok = True)

    return out_dir


# Note: collects the TR value from the image and calculates the sigma value for bandpass filtering
def calculate_sigma(image_path, hp_frequency=0.009, lp_frequency=0.08):
    import nibabel as nib
    func_img = nib.load(image_path)
    header = func_img.header
    test_tuple=header.get_zooms()
    sigma_val_hp = 1 / ((test_tuple[-1]) * hp_frequency)
    sigma_val_lp = 1 / ((test_tuple[-1]) * lp_frequency)
    return sigma_val_hp, sigma_val_lp


# Note: We want to motion correct our inputs as best as possible. To do so, we will try using the middle or first volume
# as the reference volume and use what works best. This function does just that. 
# Outputting the rms files and the par files is necessary for later steps and analyses.
def McFLIRT(in_file):
    import os
    import nipype.interfaces.fsl as fsl  # fsl
    basename = os.path.basename(in_file)
    
    RMS_THRESHOLD = 0.5 #0.5mm
    numFramesLost_0   = 0
    numFramesLost_mid = 0
    
    mcf = fsl.MCFLIRT()
    mcf.inputs.in_file    = in_file
    mcf.inputs.save_rms   = True
    mcf.inputs.save_mats  = False
    mcf.inputs.save_plots = True
    mcf.inputs.out_file = basename + '_midRef.nii.gz'
    out_mid = mcf.run()
    
    #out_mid.rms_files[0] is the abs.rms
    with open(out_mid.outputs.rms_files[0]) as f:
        for line in f:
            if float(line) > RMS_THRESHOLD:
                numFramesLost_mid +=1   

    mcf.inputs.out_file = basename + '_firstRef.nii.gz'
    mcf.inputs.ref_vol = 0
    out_first = mcf.run()
    with open(out_first.outputs.rms_files[0]) as f:
        for line in f:
            if float(line) > RMS_THRESHOLD:
                numFramesLost_0 +=1
    
    if numFramesLost_0 < numFramesLost_mid:
        print('Motion Correction with first volume as reference resulted in better alignment.')
        return out_first.outputs.out_file, out_first.outputs.par_file, out_first.outputs.rms_files
    else: 
        print('Motion Correction with middle volume as reference resulted in better alignment.')
        return out_mid.outputs.out_file, out_mid.outputs.par_file, out_mid.outputs.rms_files


# Note: calculates outliers based on framewise displacement after motion correction. 
# Input: the paramters file outputted my McFLIRT that has 6 columns indicating the 
# x,y,z translations and x,y,z rotations. Output: is a file with one line per frame
# of the BOLD. 0 indicates nonoutlier, 1 indicates an outlier. 
def CalculateOutliersFromPar(parpath):
    import os
    threshold = 0.5
    print('CalculateOutliersFromPar: input:', parpath)
    out_file = os.path.join(os.getcwd(),'fd_outliers.txt')
            
    with open(parpath, 'r') as f:
        with open(out_file, 'a') as g:
            for line in f:
                vals = line.split()
                fd = abs(float(vals[0])) + abs(float(vals[1])) + abs(float(vals[2])) +abs(float(vals[3])) +abs(float(vals[4])) +abs(float(vals[5]))
                if fd > threshold:
                    g.write('1')
                else:
                    g.write('0')
                g.write('\n')
            
    return out_file


def median_1000_normalization(in_file, mask_file=None):
    import numpy as np
    import nibabel as nib

    # Load the NIfTI image data
    img = nib.load(in_file)
    data = img.get_fdata()

    # only find the mode where there is brain tissue
    if not mask_file == None:
        print('Brain mask was provided.')
        mask_data = nib.load(mask_file).get_fdata()
        mask_4d   = mask_data[:, :, :, np.newaxis]
        numFrames = data.shape[-1]
        mask_4d   = np.tile(mask_4d, (1,1,1,numFrames))
        datamaskd = data[mask_4d == 1]
        median_value = np.median(datamaskd)
    else:
        median_value = np.median(data)

    print('Median value is {}'.format(median_value))

    # Perform mode 1000 normalization
    normalized_data = (data / median_value) * 1000

    # Create a new NIfTI image with the normalized data
    normalized_img = nib.Nifti1Image(normalized_data, img.affine, img.header)

    output_path = '{}_normalized.nii.gz'.format(in_file[:-7])
    # Save the normalized NIfTI image to the specified output path
    nib.save(normalized_img, output_path)

    return output_path


def MO_DVARS_Subprocess(in_file, mask=None):
    import os, sys
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt

    threshold = 5.

    # Load the NIfTI file
    img = nib.load(in_file)
    data = img.get_fdata()

    # Broadcast the mask to all frames
    if not mask == None:
        mask_data = nib.load(mask).get_fdata()
        mask_4d   = mask_data[:, :, :, np.newaxis]
        numFrames = data.shape[-1]
        mask_4d   = np.tile(mask_4d, (1,1,1,numFrames))
        data      = np.multiply(data, mask_4d)

    # Calculate the temporal derivative of each voxel
    diff_data = np.diff(data, axis=-1)

    # Calculate the squared difference (DVARS) per frame
    dvars = np.sqrt(np.mean((diff_data ** 2),axis=(0, 1, 2)))

    outmetricfile = 'dvars_metrics.txt'
    outmetric_path = os.path.join(os.getcwd(), outmetricfile)
    # Save DVARS values to a text file
    with open(outmetric_path, 'w') as f:
        f.write('{}\n'.format(0)) #first frame has DVARS=0
        for dvar_value in dvars:
            f.write('{}\n'.format(dvar_value))


    outfilename = 'dvars_outliers.txt'
    outfile_path = os.path.join(os.getcwd(), outfilename)
    with open(outfile_path, 'w') as f:
        f.write('{}\n'.format(0)) #first frame has DVARS=0
        for dvar_value in dvars:
            printVal = 0
            if dvar_value > threshold:
                printVal = 1
            f.write('{}\n'.format(printVal))


    outplotfile = 'dvars_plot.png'
    outplot_path = os.path.join(os.getcwd(), outplotfile)
    frames = list(range(len(dvars)))
    plt.plot(frames, dvars, linestyle='-')
    # Customize the plot
    plt.xlabel('Frames')
    plt.ylabel('DVARS Values')
    plt.title('DVARS Over Frames')

    plt.savefig(outplot_path, dpi=300, bbox_inches='tight')


    return outfile_path, outmetric_path, outplot_path


def MO_FD_Subprocess(in_file, mask):
    import subprocess
    import os, sys
    from nipype.interfaces.fsl import MotionOutliers

    outfilename = 'fd_outliers.txt'
    outmetricfile = 'fd_metrics.txt'
    outfile_path = os.path.join(os.getcwd(), outfilename)
    outmetric_path = os.path.join(os.getcwd(), outmetricfile)

    mo = MotionOutliers()
    mo.inputs.in_file = in_file
    #mo.inputs.no_motion_correction = True  ## doesn't work with FD
    #mo.inputs.mask = mask
    mo.inputs.metric = 'fd'
    mo.inputs.threshold = 0.5
    mo.inputs.out_file = outfilename
    mo.inputs.out_metric_values = outmetricfile

    ##out = mo.run()
    cmdline = mo.cmdline.split(' ')
    y = subprocess.run(cmdline)
    print(y)

    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    if outfilename not in files:
        with open(outfile_path, 'w'):
            pass
    
    return outfile_path, outmetric_path


#the artifact extraction function takes in the outliers file and the split BOLD image and removes the problematic frames
def ArtifactExtraction(split_images, art_outliers, fd_outliers):
    import os
    import numpy as np
    import json
    split_copy = split_images.copy()
    counter = 1
    fd_rejects = []

    #opens both the art_detect and dvars outlier files and creates lists of the outlier frames
    with open(art_outliers, 'r') as f:
        art_list = f.read().split()
        art_rejects = art_list.copy()
        print(art_list)
        
    #finds the problematic frames from dvars and adds them to the list of problematic frames from art_detect
    if os.stat(fd_outliers).st_size > 0:
        dvars_list = np.loadtxt(fd_outliers)
        outs = np.where(dvars_list == 1)
        output_frames = list(outs[0])
        for frame in output_frames:
            art_list.append(frame)
            fd_rejects.append(frame)
    

    #removes duplicates from the list of problematic frames
    artifact_list = [int(x) for x in art_list]
    artifact_list = list(set(artifact_list))

    #creates a dictionary with a list of the number and list of rejected frames
    reject_dict = {}
    reject_dict['Number of frames removed total'] = int(len(artifact_list))
    reject_dict['Number of frames removed by FD'] = int(len(fd_rejects))
    reject_dict['Number of frames removed by Artifact Detection'] = int(len(art_rejects))
    reject_dict['Frames rejected by FD'] = [int(x) for x in fd_rejects]
    reject_dict['Frames rejected by Artifact Detection'] = [int(x) for x in art_rejects]

    rejectionsFile = os.path.join(os.getcwd(),'rejections.json')
    with open(rejectionsFile, 'w') as r:
        json.dump(reject_dict, r, indent = 4)

    
    #removes the problematic frames from the BOLD
    for image in split_images:
        for outlier in artifact_list:
            test = '{:04d}'.format(int(outlier))
            if test in os.path.basename(image) and image in split_copy:
                split_copy.remove(image)
    return split_copy, rejectionsFile


# Note: takes in the image paths for the BOLD and template and runs the three 
# functions to output the average array, similarity matrix and mapping dictionary
def CalcSimMatrix (bold_path, template_path, maxSegVal): 
    import os
    import sys 
    import numpy as np
    import json
    sys.path.append('/data/')
    import pipeline_functions as pf
    
    
    #runs the data extraction functions
    avg_arr = pf.make_average_arr(bold_path,template_path, maxSegVal)
    sim_matrix = pf.build_sim_arr(avg_arr)
    
    #saves the extracted data files
    sim_matrix_file = os.path.join(os.getcwd(),'sim_matrix.csv')
    avg_matrix_file = os.path.join(os.getcwd(),'average_arr.csv')
    np.savetxt(sim_matrix_file, sim_matrix, delimiter=",")
    np.savetxt(avg_matrix_file, avg_arr, delimiter=",")
    
    # Note: this is not necessary and is only currently being written for backwards compatibility with later analyses
    # this can be safely removed for future projects
    mapping_dict = {i:i for i in range(0,171)}
    mapping_dict_file = os.path.join(os.getcwd(),'mapping_dict.json')
    with open(mapping_dict_file, 'w') as fp:
        json.dump(mapping_dict, fp, indent = 4)
    
    #returns the files
    return avg_matrix_file, sim_matrix_file, mapping_dict_file


def regressHeadMotion(in_file, par_file):
    import subprocess
    import os, sys
    import nipype.interfaces.fsl as fsl 

    outT2VName = "design.mat"
    outT2VPath = os.path.join(os.getcwd(),outT2VName)
    t2v = fsl.Text2Vest()
    t2v.inputs.in_file = par_file
    t2v.inputs.out_file = outT2VName
    print('The Text2Vest command is the following: '+ t2v.cmdline)
    t2v.run()

    print("\n")
    
    outResidualName = 'res4d.nii.gz'
    outResidualPath = t2v_out = os.path.join(os.getcwd(),outResidualName)
    glm = fsl.GLM(in_file=in_file, design=outT2VName, demean = True, output_type='NIFTI')
    glm.inputs.out_res_name = outResidualName
    print(glm.cmdline)
    #out = glm.run()

    cmdline = glm.cmdline.split(' ')
    y = subprocess.run(cmdline)
    print(y)
    return outResidualPath


def plotMotionMetrics(fd_metrics_file, dvars_metrics_file):
    import os
    import matplotlib.pyplot as plt 

    with open(fd_metrics_file, "r") as file1:
        data1 = [float(line.strip()) for line in file1]

    with open(dvars_metrics_file, "r") as file2:
        data2 = [float(line.strip()) for line in file2]

    # Step 2: Create a plot
    fig, ax1 = plt.subplots(figsize=(10, 6))  # Set the figure size and create primary y-axis
    fig.patch.set_facecolor('white')  # Set the figure background color to white

    # Step 3: Plot the first dataset on the primary y-axis
    time = list(range(len(data1)))
    ax1.plot(time, data1, label='Framwise Displacement', color='b', alpha=1)

    # Customize the primary y-axis
    ax1.set_xlabel('Frames')
    ax1.set_ylabel('Framwise Displacement', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a secondary y-axis
    ax2 = ax1.twinx()

    # Step 4: Plot the second dataset on the secondary y-axis
    ax2.plot(time, data2, label='DVARS', color='r', alpha=1)

    # Customize the secondary y-axis
    ax2.set_ylabel('DVARS', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Add a legend for both datasets
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)
    
    # Add thresholds recommended by Power et al 2012
    ax1.axhline(y=0.5, color='blue', linestyle='dashed', alpha=0.5)
    ax2.axhline(y=5, color='red', linestyle='dashed', alpha=0.5)

    plt.title('Motion Metrics Across Frames')
    
    outfile_path = os.path.join(os.getcwd(), 'fd_dvars_plot.png')
    plt.savefig(outfile_path, dpi=300, bbox_inches='tight')
    
    return outfile_path



# ******************************************************************************
# PIPELINE CREATION
# ******************************************************************************

#creates a pipeline
preproc = pe.Workflow(name='preproc')


#infosource iterates through the list and sends subject data into the pipeline one at a time
infosource = pe.Node(interface=util.IdentityInterface(fields=['subject']), name='infosource')
infosource.iterables = [('subject', subject_list_abs)]


#returns the directory of all input files to store outputs in
GenerateOutDir_node = pe.Node(interface=util.Function(input_names=['base_outputdir', 'image_path'], output_names=['out_dir'], function=GenerateOutDir), name='GenerateOutDir')
GenerateOutDir_node.inputs.base_outputdir = derivatives_dir
preproc.connect(infosource, 'subject', GenerateOutDir_node, 'image_path')

#the datasink node stores the outputs of all operations
datasink = pe.Node(nio.DataSink(parameterization=False), name='sinker')
preproc.connect(GenerateOutDir_node, 'out_dir', datasink, 'base_directory')


#the input node, which takes the input image from infosource and feeds it into the rest of the pipeline
input_node = pe.Node(interface=util.IdentityInterface(fields=['func']),name='input')
preproc.connect(infosource, 'subject', input_node, 'func')


#this node accesses the calculate_sigma function to take the input image and output its sigma value
sigma_value = pe.Node(interface=util.Function(input_names=['image_path', 'hp_frequency', 'lp_frequency'], output_names=['sigma_value_hp', 'sigma_value_lp'], function=calculate_sigma), name='calculate_sigmas')
sigma_value.inputs.hp_frequency=0.009
sigma_value.inputs.lp_frequency=0.08
preproc.connect(input_node, 'func', sigma_value, 'image_path')


#the template node feeds a standard brain into the linear registration node to be registered into BOLD space
template_feed = pe.Node(interface=util.IdentityInterface(fields=['template']), name='template_MNI')
template_feed.inputs.template = template_path


#the segment_feed node feeds a template segmentation into the linear registration node to be registered into BOLD space
segment_feed = pe.Node(interface=util.IdentityInterface(fields=['segment']), name='segment_AAL')
segment_feed.inputs.segment = segment_path



#the MCFLIRT node motion corrects the image
#motion_correct = pe.Node(interface=fsl.MCFLIRT(save_mats=True, save_plots=True, save_rms=True), name = 'mcflirt')
motion_correct = pe.Node(interface=util.Function(input_names=['in_file'], output_names=['out_file', 'par_file', 'rms_files'], function=McFLIRT), name='McFLIRT')
preproc.connect(input_node, 'func', motion_correct, 'in_file')


#the brain extraction node removes the extraneous organs and bones and extracts the brain from the MRI image
brain_extract = pe.Node(interface=fsl.BET(frac=0.4, mask=True, functional=True), name='bet')
preproc.connect(motion_correct, 'out_file', brain_extract, 'in_file')


#the apply bet node multiplies the brain mask to the entire BOLD image to apply the brain extraction
apply_bet = pe.Node(interface=fsl.BinaryMaths(operation = 'mul'), name = 'bet_apply')
preproc.connect(brain_extract, 'mask_file', apply_bet, 'operand_file')
preproc.connect(brain_extract, 'mask_file', datasink, OUTFOLDERNAME+'.@mask')
preproc.connect(motion_correct, 'out_file', apply_bet, 'in_file')


normalization_node = pe.Node(interface=util.Function(input_names=['in_file', 'mask_file'], output_names=['out_file'], function=median_1000_normalization), name='Median1000Normalization')
preproc.connect(apply_bet, 'out_file', normalization_node, 'in_file')
preproc.connect(brain_extract, 'mask_file', normalization_node, 'mask_file')
preproc.connect(normalization_node, 'out_file', datasink, OUTFOLDERNAME+'.@normalization')


#the artifact detection node detects any outliers in the series of MRI images and discards them
# artifact = pe.Node(interface=rpd.ArtifactDetect(norm_threshold=1.0, mask_type='file', parameter_source='FSL', zintensity_threshold=3.0), name='art_detect')
# artifact.inputs.use_norm=True
# preproc.connect(brain_extract, 'mask_file', artifact, 'mask_file')
# preproc.connect(apply_bet, 'out_file', artifact, 'realigned_files')
# preproc.connect(motion_correct, 'par_file', artifact, 'realignment_parameters')


# calculate the framewise displacement between successive frames to remove jerks
fdnode = pe.Node(interface=util.Function(input_names=['in_file', 'mask'], output_names=['outfile', 'outmetric'], function=MO_FD_Subprocess), name='fd')
preproc.connect(apply_bet, 'out_file', fdnode, 'in_file')
preproc.connect(brain_extract, 'mask_file', fdnode, 'mask')
preproc.connect(fdnode, 'outfile', datasink, OUTFOLDERNAME+'.@fd_out')
preproc.connect(fdnode, 'outmetric', datasink, OUTFOLDERNAME+'.@fd_metrics')
# ***** NOT CURRENTLY USED *****

# calculates the outliers from the transformation paramters outputted by mcflirt

# REPLACE WITH MOTION OUTLIERS AND FRAMEWISE DISPLACEMENT
# calcOutliers = pe.Node(interface=util.Function(input_names=['parpath'], output_names=['out_file'], function=CalculateOutliersFromPar), name='CalcOutliers')
# preproc.connect(motion_correct, 'par_file', calcOutliers, 'parpath')



# #the split node splits the 4D BOLD image into its contituents 3D frames to allow certain timeframes to be removed
# split = pe.Node(interface=fsl.Split(dimension='t'), name = 'splitter')
# preproc.connect(apply_bet, 'out_file', split, 'in_file')


# #the artifact extract node removes the problematic frames as indicated by the artifact detection node
# artifact_extract = pe.Node(interface=util.Function(input_names=['split_images', 'art_outliers', 'fd_outliers'], output_names=['extracted_images', 'rejectionsFile'], function=ArtifactExtraction), name='art_extract')
# preproc.connect(split, 'out_files', artifact_extract, 'split_images')
# preproc.connect(artifact, 'outlier_files', artifact_extract, 'art_outliers')
# preproc.connect(calcOutliers, 'out_file', artifact_extract, 'fd_outliers')


# #the merge node concatenates the 3D frames of the BOLD into its original 4D state after the removal of troublesome frames
# merge = pe.Node(interface=fsl.Merge(dimension = 't'), name = 'merger')
# preproc.connect(artifact_extract, 'extracted_images', merge, 'in_files')


# #the average node takes the mean of the BOLD image over time to perform bias correction
# average = pe.Node(interface=fsl.MeanImage(), name='mean_image')
# preproc.connect(merge, 'merged_file', average, 'in_file')


# #the bias correct node takes the average frame of the BOLD and outputs a bias field that can be used for all other frames
# bias_correct = pe.Node(interface=fsl.FAST(bias_iters=2, output_biascorrected=True, output_biasfield=True), name='bias_correction')
# preproc.connect(average, 'out_file', bias_correct, 'in_files')


# #the apply bias node subtracts the bias field from the entire BOLD image to apply the bias correction
# apply_bias = pe.Node(interface=fsl.BinaryMaths(operation = 'sub'), name = 'bias_apply')
# preproc.connect(bias_correct, 'bias_field', apply_bias, 'operand_file')
# preproc.connect(merge, 'merged_file', apply_bias, 'in_file')

#the smoothing node smooths the BOLD image. The 6mm fwhm informed by Power et al.
smooth = pe.Node(interface=fsl.Smooth(), name='smoothing')
smooth.inputs.fwhm = 6.0
preproc.connect(normalization_node, 'out_file', smooth, 'in_file')

#eventually smooth after bias correction
#preproc.connect(apply_bias, 'out_file', smooth, 'in_file')


#the bandpass filtering node filters out extraneous frequencies from the MRI image
band_pass = pe.Node(interface=fsl.TemporalFilter(), name='bandpass_filtering')
preproc.connect(sigma_value, 'sigma_value_hp', band_pass, 'highpass_sigma')
preproc.connect(sigma_value, 'sigma_value_lp', band_pass, 'lowpass_sigma')
preproc.connect(smooth, 'smoothed_file', band_pass, 'in_file')

#this node will regress away the headmotion parameters and return the residuals
regressNode = pe.Node(interface=util.Function(input_names=['in_file', 'par_file'], output_names=['out_file'], function=regressHeadMotion), name='RegressMotionParameters')
preproc.connect(band_pass, 'out_file', regressNode, 'in_file')
preproc.connect(motion_correct, 'par_file', regressNode, 'par_file')
preproc.connect(regressNode, 'out_file', datasink, OUTFOLDERNAME+'.@residual_out')


dvarsnode = pe.Node(interface=util.Function(input_names=['in_file', 'mask'], output_names=['outfile', 'outmetric', 'outplot_path'], function=MO_DVARS_Subprocess), name='dvars')
#preproc.connect(normalization_node2, 'out_file', dvarsnode, 'in_file')
preproc.connect(regressNode, 'out_file', dvarsnode, 'in_file')
preproc.connect(brain_extract, 'mask_file', dvarsnode, 'mask')
preproc.connect(dvarsnode, 'outfile', datasink, OUTFOLDERNAME+'.@dvars_out')
preproc.connect(dvarsnode, 'outmetric', datasink, OUTFOLDERNAME+'.@dvars_metrics')
preproc.connect(dvarsnode, 'outplot_path', datasink, OUTFOLDERNAME+'.@dvars_plot')


plotmotionmetrics_node = pe.Node(interface=util.Function(input_names=['fd_metrics_file', 'dvars_metrics_file'], output_names=['outfile_path'], function=plotMotionMetrics), name='plot_fd_vs_dvars')
preproc.connect(fdnode, 'outmetric', plotmotionmetrics_node, 'fd_metrics_file')
preproc.connect(dvarsnode, 'outmetric', plotmotionmetrics_node, 'dvars_metrics_file')
preproc.connect(plotmotionmetrics_node, 'outfile_path', datasink, OUTFOLDERNAME+'.@fdvsdvars_plot')

# #the linear registration node registers the standard brain into BOLD space using the BOLD image as reference and only using linear registration
# lin_reg = pe.Node(interface=fsl.FLIRT(), name='linear_reg')
# preproc.connect(merge, 'merged_file', lin_reg, 'reference')
# preproc.connect(template_feed, 'template', lin_reg, 'in_file')


# #the non-linear registration node registers the linear registered brain to match the BOLD image using non-linear registration
# non_reg = pe.Node(interface=fsl.FNIRT(), name='nonlinear_reg')
# non_reg.inputs.in_fwhm            = [8, 4, 2, 2]
# non_reg.inputs.subsampling_scheme = [4, 2, 1, 1]
# non_reg.inputs.warp_resolution    = (6, 6, 6)
# non_reg.inputs.max_nonlin_iter    = [1,1,1,1]
# preproc.connect(lin_reg, 'out_file', non_reg, 'in_file')
# preproc.connect(merge, 'merged_file', non_reg, 'ref_file')


# #the apply_lin node applies the same linear registration as the standard brain to the template segmentation
# apply_lin = pe.Node(interface=fsl.ApplyXFM(interp='nearestneighbour'), name='apply_linear')
# preproc.connect(segment_feed, 'segment', apply_lin, 'in_file')
# preproc.connect(merge, 'merged_file', apply_lin, 'reference')
# preproc.connect(lin_reg, 'out_matrix_file', apply_lin, 'in_matrix_file')


# #the apply_non node applies the same non-linear registration as the standard brain to the template segmentation
# apply_non = pe.Node(interface=fsl.ApplyWarp(interp='nn'), name='apply_nonlin')
# preproc.connect(apply_lin, 'out_file', apply_non, 'in_file')
# preproc.connect(merge, 'merged_file', apply_non, 'ref_file')
# preproc.connect(non_reg, 'field_file', apply_non, 'field_file')

# rename_node = pe.Node(interface=util.Rename(), name='Rename')
# rename_node.inputs.keep_ext = True
# rename_node.inputs.format_string = 'final_preprocessed_output'
# preproc.connect(smooth, 'smoothed_file',rename_node, 'in_file')



# #the data extraction node takes in the BOLD and template images and extracts the necessary data (average voxel intensity per region, a similarity matrix, and a mapping dictionary)
# CalcSimMatrix_node = pe.Node(interface=util.Function(input_names=['bold_path', 'template_path', 'maxSegVal'], output_names=['avg_arr_file', 'sim_matrix_file', 'mapping_dict_file'], function=CalcSimMatrix), name='CalcSimMatrix')
# CalcSimMatrix_node.inputs.maxSegVal = MAX_SEGMENT_VAL
# preproc.connect(smooth, 'smoothed_file', CalcSimMatrix_node, 'bold_path')
# preproc.connect(apply_non, 'out_file', CalcSimMatrix_node, 'template_path')
# preproc.connect(CalcSimMatrix_node, 'avg_arr_file', datasink, OUTFOLDERNAME+'.@avgBoldSigPerRegion')
# preproc.connect(CalcSimMatrix_node, 'sim_matrix_file', datasink, OUTFOLDERNAME+'.@similarityMatrix')
# preproc.connect(CalcSimMatrix_node, 'mapping_dict_file', datasink, OUTFOLDERNAME+'.@MappingDict')


# ******************************************************************************
# IF MEMORY IS PLENTIFUL, THEN SAVE EVERYTHING
# if(SAVE_INTERMEDIATES):
#     preproc.connect(segment_feed, 'segment', datasink, OUTFOLDERNAME+'.@OGSeg')
#     preproc.connect(motion_correct, 'out_file', datasink, OUTFOLDERNAME+'.@mcf_out')
#     preproc.connect(motion_correct, 'par_file', datasink, OUTFOLDERNAME+'.@mcf_par')
#     preproc.connect(motion_correct, 'rms_files', datasink, OUTFOLDERNAME+'.@mcf_rms')
#     preproc.connect(brain_extract, 'out_file', datasink, OUTFOLDERNAME+'.@be_out')
#     preproc.connect(apply_bet, 'out_file', datasink, OUTFOLDERNAME+'.@applybe_out')
#     preproc.connect(artifact, 'outlier_files', datasink, OUTFOLDERNAME+'.@artdet_outs')
#     preproc.connect(calcOutliers, 'out_file', datasink, OUTFOLDERNAME+'.@calcFDOuts_outs')
#     preproc.connect(artifact_extract, 'rejectionsFile', datasink, OUTFOLDERNAME+'.@rejects_summ')
#     preproc.connect(merge, 'merged_file', datasink, OUTFOLDERNAME+'.@merge_out')
#     preproc.connect(bias_correct, 'bias_field', datasink, OUTFOLDERNAME+'.@bias')
#     preproc.connect(apply_bias, 'out_file', datasink, OUTFOLDERNAME+'.@appbias_out')
#     preproc.connect(band_pass, 'out_file', datasink, OUTFOLDERNAME+'.@bandpass_out')
#     preproc.connect(smooth, 'smoothed_file', datasink, OUTFOLDERNAME+'.@smooth_out')
#     preproc.connect(lin_reg, 'out_file', datasink, OUTFOLDERNAME+'.@lin_out')
#     preproc.connect(lin_reg, 'out_matrix_file', datasink, OUTFOLDERNAME+'.@lin_mat')
#     preproc.connect(non_reg, 'warped_file', datasink, OUTFOLDERNAME+'.@nlin_out')
#     preproc.connect(non_reg, 'field_file', datasink, OUTFOLDERNAME+'.@nlin_mat')
#     preproc.connect(apply_lin, 'out_file', datasink, OUTFOLDERNAME+'.@app_lin_out')
#     preproc.connect(apply_non, 'out_file', datasink, OUTFOLDERNAME+'.@app_nlin_out')
#     preproc.connect(rename_node, 'out_file',datasink, OUTFOLDERNAME+'.@final_out')
# ******************************************************************************

# ******************************************************************************
# PIPELINE RUN
# ******************************************************************************

#creates a workflow diagram (IN THE CURRENT WORKING DIRECTORY)
preproc.write_graph()

preproc.run()
