################################################################################
# Author:  Joy Roy, William Reynolds, Rafael Ceschin
# Purpose: The following is a pipeline made by PIRC at UPMC CHP in order to
#          preprocess BOLD fMRIs and to use those outputs for similarity 
#          matrix generation.
#
#
# Contact: jor115@pitt.edu
# Acknowledgments: Ansh Patel from The Hillman Academy contributed to this work.
################################################################################
import argparse
import nibabel as nib
import nipy as nipy
import nipype.algorithms.rapidart as rpd    # rapidart module
import nipype.interfaces.io as nio          # Data i/o
import nipype.interfaces.fsl as fsl         # fsl
import nipype.interfaces.ants as ants       # ANTs
import nipype.interfaces.utility as util    # utility
import nipype.pipeline.engine as pe         # pypeline engine
import numpy as np
import os, sys
import time


DATATYPE_SUBJECT_DIR = 'func'
DATATYPE_FILE_SUFFIX = 'bold' 
SAVE_INTERMEDIATES = True
scheduleTXT   = '/app/Template/sched.txt'


def makeParser():
    parser = argparse.ArgumentParser(
                        prog='Sim_Funky_Pipeline', 
                        usage='This program preprocesses fMRIs for later use with connectivity analyses',
                        epilog='BUG REPORTING: Report bugs to pirc@chp.edu or more directly to Joy Roy at the Childrens Hospital of Pittsburgh.'
        )
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-p','--parentDir', nargs=1, required=True,
                        help='Path to the parent data directory. BIDS compatible datasets are encouraged.')
    parser.add_argument('-sid','--subject_id', nargs=1, required=True,
                        help='Subject ID used to indicate which patient to preprocess')
    parser.add_argument('-spath','--subject_t1_path', nargs=1, required=False,
                        help='Path to a subjects T1 scan. This is not necessary if subject ID is provided as the T1 will be automatically found using the T1w.nii.gz extension')
    parser.add_argument('-ses_id','--session_id', nargs=1, required=False,
                        help='Session ID used to indicate which session to look for the patient to preprocess')
    parser.add_argument('-tem','--template', nargs=1, required=False,
                        help='Template to be used to register into patient space. Default is MNI152lin_T1_2mm_brain.nii.gz')
    parser.add_argument('-seg','--segment', nargs=1, required=False,
                        help='Atlas to be used to identify brain regions in patient space. This is used in conjunction with the template. Please ensure that the atlas is in the same space as the template. Default is the AALv3 template.')
    parser.add_argument('-o','--ourDir', nargs=1, required=True,
                        help='Path to the \'derivatives\' folder or chosen out folder. All results will be submitted to outDir/out/str_preproc/subject_id/...')
    parser.add_argument('--saveIntermediates', required=False, action='store_true',
                        help='Saves all intermediate files while pipeline is running.')
    parser.add_argument('--testmode', required=False, action='store_true',
                        help='Activates TEST_MODE to make pipeline finish faster for quicker debugging')

    return parser 

# This was developed instead of using the default parameter in the argparser
# bc argparser only returns a list or None and you can't do None[0]. 
# Not all variables need a default but need to be inspected whether they are None
def vetArgNone(variable, default):
    if variable==None:
        return default
    else:
        return variable[0]

def makeOutDir(outDirName, args, enforceBIDS=True):
    outDir = ''
    if os.path.basename(args.ourDir[0]) == 'derivatives':
        outDir = os.path.join(args.ourDir[0], outDirName, args.subject_id[0])
    elif args.ourDir[0] == args.parentDir[0]:
        print("Your outdir is the same as your parent dir!")
        print("Making a derivatives folder for you...")
        outDir = os.path.join(args.ourDir[0], 'derivatives', outDirName, args.subject_id[0])
    elif os.path.basename(args.ourDir[0]) == args.subject_id[0]:
        print('The given out directory seems to be at a patient level rather than parent level')
        print('It is hard to determine if your out directory is BIDS compliant')
    elif 'derivatives' in args.ourDir[0]:
        outDir = os.path.join(args.ourDir[0], outDirName, args.subject_id[0])

    if not os.path.exists(outDir):
        os.makedirs(outDir, exist_ok=True)

    return outDir


# Note: collects the TR value from the image and calculates the sigma value for bandpass filtering
def calculate_sigma(image_path, hp_frequency=0.009, lp_frequency=0.08):
    import nibabel as nib
    func_img = nib.load(image_path)
    header = func_img.header
    test_tuple=header.get_zooms()
    sigma_val_hp = 1 / ((test_tuple[-1]) * hp_frequency)
    sigma_val_lp = 1 / ((test_tuple[-1]) * lp_frequency)
    return sigma_val_hp, sigma_val_lp


# Nipype nodes built on python-functions need to reimport libraries seperately
def getMaxROI(atlas_path):
    import nibabel as nib
    import numpy as np
    
    img = nib.load(atlas_path)
    data = img.get_fdata()
    return round(np.max(data))


# Note: This function helps to determine the best volume to use as a reference for motion correction.
def findBestReference(in_file, scheduleTXT, derivatives_dir):
    import nibabel as nib
    import numpy as np
    from tqdm import tqdm
    import os, sys
    import json
    sys.path.append('/data/')
    import pipeline_functions as pf

    entryname = os.path.basename(in_file)
    file_name = "best_frames.json"
    # Check if the file exists in the directory
    bestFramesfile_path = os.path.join(derivatives_dir, file_name)
    if os.path.exists(bestFramesfile_path):

        with open(bestFramesfile_path, 'r') as json_file:
            data_dict = json.load(json_file)

        print('A cached file containing the best frames of several scans already exists: {}'.format(bestFramesfile_path))

        if entryname in data_dict.keys():
            print('The best frame for this file was previously calculated and will be used now.')
            return data_dict[entryname], bestFramesfile_path

    else:
        print('A best frames cache file does not exist. It will be made now.')
        bestFramesfile_path = os.path.join(os.getcwd(), file_name)
        with open(bestFramesfile_path, 'w') as file:
            file.write("{}")
        print("File created.")
        data_dict = {}

    img = nib.load(in_file)
    numFrames = img.get_fdata().shape[-1]
    matrix = np.zeros((numFrames,numFrames))

    roi_basename = os.path.basename(in_file)[:-7] + '_vi'
    print('Note: the first iteration will take the longest.')
    for i in tqdm(range(numFrames)):

        v0 =  '{}{}.nii.gz'.format(roi_basename, i)
        if not os.path.exists(v0):
            v0 = pf.getVolume(in_file,i, v0)

        for j in range(i, numFrames): 
            v1 = '{}{}.nii.gz'.format(roi_basename, j)
            if not os.path.exists(v1):
                v1 = pf.getVolume(in_file,j, v1)

            sim = pf.getSimilarityBetweenVolumes(v0, v1, scheduleTXT)
            matrix[i,j] = sim
            matrix[j,i] = sim
    
    # clear temporary files in consideration for storage
    for filename in os.listdir('.'):
        if filename.startswith(roi_basename):
            os.remove(filename)

    column_means = np.mean(matrix, axis=0)
    bestVol = np.argmin(column_means).item()
    print('Volume number {} was identified as the best reference for motion correction.'.format(bestVol))

    data_dict[entryname] = bestVol

    print("This calculation will be saved in cache...")
    with open(bestFramesfile_path, 'w') as json_file:
        json.dump(data_dict, json_file)

    return bestVol, bestFramesfile_path


# Note: This function is used normalize the median of the data to 1000
#       Power et al normalized their mode to 1000, but we believe median is more stable.
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


# Note: This function is used to calculate the DVARS values across the scan
# MOtion_DVARS_Subprocess
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
def ArtifactExtraction(split_images, dvars_outliers, fd_outliers):
    import os
    import numpy as np
    import json
    split_copy = split_images.copy()
    counter = 1
    fd_rejects = []
    dvars_rejects = []

    #opens both the art_detect and dvars outlier files and creates lists of the outlier frames
    if os.stat(dvars_outliers).st_size > 0:
        dvars_list = np.loadtxt(dvars_outliers)
        outs = np.where(dvars_list == 1)
        output_frames = list(outs[0])
        for frame in output_frames:
            dvars_rejects.append(frame)

        
    #finds the problematic frames from dvars and adds them to the list of problematic frames from art_detect
    if os.stat(fd_outliers).st_size > 0:
        fd_list = np.loadtxt(fd_outliers)
        outs = np.where(dvars_list == 1)
        output_frames = list(outs[0])
        for frame in output_frames:
            fd_rejects.append(frame)
    

    #removes duplicates from the list of problematic frames
    all_rejects = list(set(fd_rejects).union(dvars_rejects))

    #creates a dictionary with a list of the number and list of rejected frames
    reject_dict = {}
    reject_dict['Number of frames removed total'] = int(len(all_rejects))
    reject_dict['Number of frames removed by FD'] = int(len(fd_rejects))
    reject_dict['Number of frames removed by DVARS'] = int(len(dvars_rejects))
    reject_dict['Frames rejected by FD'] = [int(x) for x in set(fd_rejects)]
    reject_dict['Frames rejected by DVARS'] = [int(x) for x in set(dvars_rejects)]

    rejectionsFile = os.path.join(os.getcwd(),'rejections.json')
    with open(rejectionsFile, 'w') as r:
        json.dump(reject_dict, r, indent = 4)

    
    #removes the problematic frames from the BOLD
    for image in split_images:
        for outlier in all_rejects:
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
    mapping_dict = {i:i for i in range(0,maxSegVal+1)}
    mapping_dict_file = os.path.join(os.getcwd(),'mapping_dict.json')
    with open(mapping_dict_file, 'w') as fp:
        json.dump(mapping_dict, fp, indent = 4)
    
    #returns the files
    return avg_matrix_file, sim_matrix_file, mapping_dict_file

# Note: This function expands the original 6 motion parameters to 24 (R R**2 R' R'**2)
def expandMotionParameters(par_file):
    import numpy as np
    import os
    
    # Read the original .par file
    original_params = np.loadtxt(par_file)

    # Calculate the additional parameters
    squared_params = original_params**2
    derivatives = np.diff(original_params, axis=0)
    # Add one row back in
    derivatives = np.vstack((np.zeros(original_params.shape[1]), derivatives))
    squared_derivatives = derivatives**2

    # Combine all parameters
    expanded_params = np.hstack((original_params, squared_params, derivatives, squared_derivatives))

    # Write the expanded parameters to a new .par file
    outfile = os.path.join(os.getcwd(), 'expanded.par')
    np.savetxt(outfile, expanded_params, fmt='%.10g')
    
    return outfile


def regressHeadMotion(in_file, par_file):
    import subprocess
    import os, sys
    import nipype.interfaces.fsl as fsl 

    outT2VName = "design.mat"
    outT2VPath = os.path.join(os.getcwd(),outT2VName)
    t2v = fsl.Text2Vest()
    t2v.inputs.in_file = par_file
    t2v.inputs.out_file = outT2VName
    print(t2v.cmdline)
    t2v.run()
    
    outResidualName = 'res4d.nii.gz'
    outResidualPath = t2v_out = os.path.join(os.getcwd(),outResidualName)
    glm = fsl.GLM(in_file=in_file, design=outT2VName, demean = True, output_type='NIFTI')
    glm.inputs.out_res_name = outResidualName
    print(glm.cmdline)
    #out = glm.run()

    cmdline = glm.cmdline.split(' ')
    y = subprocess.run(cmdline)
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

def buildWorkflow(patient_func_path, template_path, segment_path, outDir, subjectID, testmode=False, saveIntermediates=False):
    #creates a pipeline
    preproc = pe.Workflow(name='preproc')

    #the input node, which takes the input image from infosource and feeds it into the rest of the pipeline
    input_node = pe.Node(interface=util.IdentityInterface(fields=['func']),name='input')
    input_node.inputs.func = patient_func_path


    #the datasink node stores the outputs of all operations
    datasink = pe.Node(nio.DataSink(parameterization=False), name='sinker')
    datasink.inputs.base_directory = outDir


    reorient2std_node = pe.Node(interface=fsl.Reorient2Std(), name='reorient2std')
    preproc.connect(input_node, 'func', reorient2std_node, 'in_file')
    preproc.connect(reorient2std_node, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@reorient')


    #this node accesses the calculate_sigma function to take the input image and output its sigma value
    sigma_value = pe.Node(interface=util.Function(input_names=['image_path', 'hp_frequency', 'lp_frequency'], output_names=['sigma_value_hp', 'sigma_value_lp'], function=calculate_sigma), name='calculate_sigmas')
    sigma_value.inputs.hp_frequency=0.009
    sigma_value.inputs.lp_frequency=0.08
    preproc.connect(reorient2std_node, 'out_file', sigma_value, 'image_path')


    #the template node feeds a standard brain into the linear registration node to be registered into BOLD space
    template_feed = pe.Node(interface=util.IdentityInterface(fields=['template']), name='template_MNI')
    template_feed.inputs.template = template_path


    #the segment_feed node feeds a template segmentation into the linear registration node to be registered into BOLD space
    segment_feed = pe.Node(interface=util.IdentityInterface(fields=['segment']), name='segment_AAL')
    segment_feed.inputs.segment = segment_path

    # # finds the best frame to use as a reference
    bestRef_node = pe.Node(interface=util.Function(input_names=['in_file', 'scheduleTXT', 'derivatives_dir'], output_names=['bestReference', 'bestFramesFile'], function=findBestReference), name='findBestReference')
    bestRef_node.inputs.scheduleTXT = scheduleTXT
    bestRef_node.inputs.derivatives_dir = os.path.join(datasink.inputs.base_directory, DATATYPE_SUBJECT_DIR)
    preproc.connect(input_node, 'func', bestRef_node, 'in_file')

    #the MCFLIRT node motion corrects the image
    motion_correct = pe.Node(interface=fsl.MCFLIRT(save_plots = True, save_rms= True), name='McFLIRT')
    preproc.connect(reorient2std_node, 'out_file', motion_correct, 'in_file')
    preproc.connect(bestRef_node, 'bestReference', motion_correct, 'ref_vol')

    fslroi_node_2 = pe.Node(interface=fsl.ExtractROI(t_size=1), name = 'extractRoi_2')
    preproc.connect(motion_correct, 'out_file', fslroi_node_2, 'in_file')
    preproc.connect(bestRef_node, 'bestReference', fslroi_node_2, 't_min')

    #the brain extraction node removes the nonbrain tissue and extracts the brain from the MRI image
    brain_extract = pe.Node(interface=fsl.BET(frac=0.45, mask=True, robust=True), name='bet')
    # functional=True,
    preproc.connect(fslroi_node_2, 'roi_file', brain_extract, 'in_file')


    #the apply bet node multiplies the brain mask to the entire BOLD image to apply the brain extraction
    apply_bet = pe.Node(interface=fsl.BinaryMaths(operation = 'mul'), name = 'bet_apply')
    preproc.connect(brain_extract, 'mask_file', apply_bet, 'operand_file')
    preproc.connect(motion_correct, 'out_file', apply_bet, 'in_file')


    # we normalize the brain to 1000 as recommended by Power et al, however we normalize to median instead of the mode
    normalization_node = pe.Node(interface=util.Function(input_names=['in_file', 'mask_file'], output_names=['out_file'], function=median_1000_normalization), name='Median1000Normalization')
    preproc.connect(apply_bet, 'out_file', normalization_node, 'in_file')
    preproc.connect(brain_extract, 'mask_file', normalization_node, 'mask_file')


    # calculate the framewise displacement between successive frames to remove jerks
    fdnode = pe.Node(interface=util.Function(input_names=['in_file', 'mask'], output_names=['outfile', 'outmetric'], function=MO_FD_Subprocess), name='fd')
    preproc.connect(apply_bet, 'out_file', fdnode, 'in_file')
    preproc.connect(brain_extract, 'mask_file', fdnode, 'mask')


    # expand 6 motion parameters to 24
    expandParNode = pe.Node(interface=util.Function(input_names=['par_file'], output_names=['out_file'], function=expandMotionParameters), name='ExpandMotionParameters')
    preproc.connect(motion_correct, 'par_file', expandParNode, 'par_file')

    #this node will regress away the headmotion parameters and return the residuals
    regressNode = pe.Node(interface=util.Function(input_names=['in_file', 'par_file'], output_names=['out_file'], function=regressHeadMotion), name='RegressMotionParameters')
    preproc.connect(normalization_node, 'out_file', regressNode, 'in_file')
    preproc.connect(expandParNode, 'out_file', regressNode, 'par_file')


    #the bandpass filtering node filters out extraneous frequencies from the MRI image
    band_pass = pe.Node(interface=fsl.TemporalFilter(), name='bandpass_filtering')
    preproc.connect(sigma_value, 'sigma_value_hp', band_pass, 'highpass_sigma')
    preproc.connect(sigma_value, 'sigma_value_lp', band_pass, 'lowpass_sigma')
    preproc.connect(regressNode, 'out_file', band_pass, 'in_file')

    #the smoothing node smooths the BOLD image. The 6mm fwhm informed by Power et al.
    smooth = pe.Node(interface=fsl.Smooth(), name='smoothing')
    smooth.inputs.fwhm = 6.0
    preproc.connect(band_pass, 'out_file', smooth, 'in_file')


    #a custom function to calculate dvars as indicated by Power et al. We noticed that FSL's motionoutlier renormalized before calculating dvars, which is not desirable here
    dvarsnode = pe.Node(interface=util.Function(input_names=['in_file', 'mask'], output_names=['outfile', 'outmetric', 'outplot_path'], function=MO_DVARS_Subprocess), name='dvars')
    preproc.connect(smooth, 'smoothed_file', dvarsnode, 'in_file')
    preproc.connect(brain_extract, 'mask_file', dvarsnode, 'mask')


    # a custom function to plot dvars values against fd values
    plotmotionmetrics_node = pe.Node(interface=util.Function(input_names=['fd_metrics_file', 'dvars_metrics_file'], output_names=['outfile_path'], function=plotMotionMetrics), name='plot_fd_vs_dvars')
    preproc.connect(fdnode, 'outmetric', plotmotionmetrics_node, 'fd_metrics_file')
    preproc.connect(dvarsnode, 'outmetric', plotmotionmetrics_node, 'dvars_metrics_file')


    #the split node splits the 4D BOLD image into its contituents 3D frames to allow certain timeframes to be removed
    split = pe.Node(interface=fsl.Split(dimension='t'), name = 'splitter')
    preproc.connect(smooth, 'smoothed_file', split, 'in_file')


    #the artifact extract node removes the problematic frames as indicated by the artifact detection node
    artifact_extract = pe.Node(interface=util.Function(input_names=['split_images', 'dvars_outliers', 'fd_outliers'], output_names=['extracted_images', 'rejectionsFile'], function=ArtifactExtraction), name='art_extract')
    preproc.connect(split, 'out_files', artifact_extract, 'split_images')
    preproc.connect(dvarsnode, 'outfile', artifact_extract, 'dvars_outliers')
    preproc.connect(fdnode, 'outfile', artifact_extract, 'fd_outliers')


    #the merge node concatenates the 3D frames of the BOLD into its original 4D state after the removal of troublesome frames
    merge = pe.Node(interface=fsl.Merge(dimension = 't'), name = 'merger')
    preproc.connect(artifact_extract, 'extracted_images', merge, 'in_files')

    fslroi_node = pe.Node(interface=fsl.ExtractROI(t_size=1), name = 'extractRoi')
    preproc.connect(apply_bet, 'out_file', fslroi_node, 'in_file')
    preproc.connect(bestRef_node, 'bestReference', fslroi_node, 't_min')


    # ants for both linear and nonlinear registration
    antsReg = pe.Node(interface=ants.Registration(), name='antsRegistration')
    antsReg.inputs.transforms = ['Affine', 'SyN']
    antsReg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    antsReg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    if testmode==True:
        antsReg.inputs.number_of_iterations = [[5, 5], [5, 5, 5]]
    antsReg.inputs.dimension = 3
    antsReg.inputs.write_composite_transform = False
    antsReg.inputs.collapse_output_transforms = False
    antsReg.inputs.initialize_transforms_per_stage = False
    antsReg.inputs.metric = ['Mattes']*2
    antsReg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    antsReg.inputs.radius_or_number_of_bins = [32]*2
    antsReg.inputs.sampling_strategy = ['Random', None]
    antsReg.inputs.sampling_percentage = [0.05, None]
    antsReg.inputs.convergence_threshold = [1.e-8, 1.e-9]
    antsReg.inputs.convergence_window_size = [20]*2
    antsReg.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
    antsReg.inputs.sigma_units = ['vox'] * 2
    antsReg.inputs.shrink_factors = [[2,1], [3,2,1]]
    antsReg.inputs.use_histogram_matching = [True, True] # This is the default
    antsReg.inputs.output_warped_image = 'output_warped_image.nii.gz'

    preproc.connect(template_feed, 'template', antsReg, 'moving_image')
    preproc.connect(fslroi_node, 'roi_file', antsReg, 'fixed_image')

    antsAppTrfm = pe.Node(interface=ants.ApplyTransforms(), name='antsApplyTransform')
    antsAppTrfm.inputs.dimension = 3
    antsAppTrfm.inputs.interpolation = 'NearestNeighbor'
    antsAppTrfm.inputs.default_value = 0

    preproc.connect(segment_feed, 'segment', antsAppTrfm, 'input_image')
    preproc.connect(fslroi_node, 'roi_file', antsAppTrfm, 'reference_image')
    preproc.connect(antsReg, 'reverse_forward_transforms', antsAppTrfm, 'transforms')
    preproc.connect(antsReg, 'reverse_forward_invert_flags', antsAppTrfm, 'invert_transform_flags')

    rename_node = pe.Node(interface=util.Rename(), name='Rename')
    rename_node.inputs.keep_ext = True
    rename_node.inputs.format_string = 'final_preprocessed_output'
    preproc.connect(merge, 'merged_file',rename_node, 'in_file')
    preproc.connect(rename_node, 'out_file',datasink, DATATYPE_SUBJECT_DIR+'.@final_out')

    GetMaxROI_node = pe.Node(interface=util.Function(input_names=['atlas_path'], output_names=['max_roi'], function=getMaxROI), name='GetMaxROI')
    preproc.connect(segment_feed, 'segment', GetMaxROI_node, 'atlas_path')

    #the data extraction node takes in the BOLD and template images and extracts the necessary data (average voxel intensity per region, a similarity matrix, and a mapping dictionary)
    CalcSimMatrix_node = pe.Node(interface=util.Function(input_names=['bold_path', 'template_path', 'maxSegVal'], output_names=['avg_arr_file', 'sim_matrix_file', 'mapping_dict_file'], function=CalcSimMatrix), name='CalcSimMatrix')
    preproc.connect(GetMaxROI_node, 'max_roi', CalcSimMatrix_node, 'maxSegVal')
    preproc.connect(merge, 'merged_file', CalcSimMatrix_node, 'bold_path')
    preproc.connect(antsAppTrfm, 'output_image', CalcSimMatrix_node, 'template_path') # FSL Registation implementation
    

    # Should always be outputted
    preproc.connect(bestRef_node, 'bestFramesFile', datasink, '{}.@bestFramesFile'.format(DATATYPE_SUBJECT_DIR))
    preproc.connect(antsReg, 'warped_image', datasink, '{}.@warpedTemplate'.format(DATATYPE_SUBJECT_DIR))
    preproc.connect(antsAppTrfm, 'output_image', datasink, '{}.@warpedAtlas'.format(DATATYPE_SUBJECT_DIR))
    preproc.connect(CalcSimMatrix_node, 'avg_arr_file', datasink, DATATYPE_SUBJECT_DIR+'.@avgBoldSigPerRegion')
    preproc.connect(CalcSimMatrix_node, 'sim_matrix_file', datasink, DATATYPE_SUBJECT_DIR+'.@similarityMatrix')
    preproc.connect(plotmotionmetrics_node, 'outfile_path', datasink, DATATYPE_SUBJECT_DIR+'.@fdvsdvars_plot')



    # # ******************************************************************************
    # # IF MEMORY IS PLENTIFUL, THEN SAVE EVERYTHING
    if(saveIntermediates):
        preproc.connect(segment_feed, 'segment', datasink, DATATYPE_SUBJECT_DIR+'.@OGSeg')
        preproc.connect(motion_correct, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@mcf_out')
        preproc.connect(motion_correct, 'par_file', datasink, DATATYPE_SUBJECT_DIR+'.@mcf_par')
        preproc.connect(motion_correct, 'rms_files', datasink, DATATYPE_SUBJECT_DIR+'.@mcf_rms')
        preproc.connect(brain_extract, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@be_out')
        preproc.connect(apply_bet, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@applybe_out')
        preproc.connect(normalization_node, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@normalization')
        preproc.connect(artifact, 'outlier_files', datasink, DATATYPE_SUBJECT_DIR+'.@artdet_outs')
        preproc.connect(calcOutliers, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@calcFDOuts_outs')
        preproc.connect(artifact_extract, 'rejectionsFile', datasink, DATATYPE_SUBJECT_DIR+'.@rejects_summ')
        preproc.connect(merge, 'merged_file', datasink, DATATYPE_SUBJECT_DIR+'.@merge_out')
        preproc.connect(bias_correct, 'bias_field', datasink, DATATYPE_SUBJECT_DIR+'.@bias')
        preproc.connect(regressNode, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@residual_out')
        preproc.connect(apply_bias, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@appbias_out')
        preproc.connect(band_pass, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@bandpass_out')
        preproc.connect(smooth, 'smoothed_file', datasink, DATATYPE_SUBJECT_DIR+'.@smooth_out')
        preproc.connect(antsAppTrfm, 'out_file', datasink, DATATYPE_SUBJECT_DIR+'.@app_nlin_out')
        preproc.connect(fdnode, 'outfile', datasink, DATATYPE_SUBJECT_DIR+'.@fd_out')
        preproc.connect(fdnode, 'outmetric', datasink, DATATYPE_SUBJECT_DIR+'.@fd_metrics')
        preproc.connect(dvarsnode, 'outfile', datasink, DATATYPE_SUBJECT_DIR+'.@dvars_out')
        preproc.connect(dvarsnode, 'outmetric', datasink, DATATYPE_SUBJECT_DIR+'.@dvars_metrics')
        preproc.connect(dvarsnode, 'outplot_path', datasink, DATATYPE_SUBJECT_DIR+'.@dvars_plot')
        preproc.connect(CalcSimMatrix_node, 'mapping_dict_file', datasink, DATATYPE_SUBJECT_DIR+'.@MappingDict')
    # # ******************************************************************************

    return preproc





def main():
    parser = makeParser()
    args   = parser.parse_args()
    data_dir      = args.parentDir[0]
    outDir        = ''
    outDirName    = 'Sim_Funky_Pipeline'
    session       = vetArgNone(args.session_id, None)
    template_path = vetArgNone(args.template, '/app/Template/MNI152lin_T1_2mm_brain.nii.gz') #path in docker container
    segment_path  = vetArgNone(args.segment, '/app/Template/AAL3v1_CombinedThalami.nii.gz') #path in docker container
    enforceBIDS   = True
    outDir        = makeOutDir(outDirName, args, enforceBIDS)

    if args.testmode:
        print("!!YOU ARE USING TEST MODE!!")

    for i in os.listdir(args.parentDir[0]):
        if i[:3] == 'ses':
            if session == None:
                raise Exception("Your data is sorted into sessions but you did not indicate a session to process. Please provide the Session.")

    if session != None:
        patient_func_dir = os.path.join(args.parentDir[0], session, args.subject_id[0], DATATYPE_SUBJECT_DIR)
    else:
        patient_func_dir = os.path.join(args.parentDir[0], args.subject_id[0], DATATYPE_SUBJECT_DIR)

    ## The following behavior only takes the first T1 seen in the directory. 
    ## The code could be expanded to account for multiple runs
    patient_func_path = None
    for i in os.listdir(patient_func_dir):
        if i[-11:] =='{}.nii.gz'.format(DATATYPE_FILE_SUFFIX):
            patient_func_path = os.path.join(patient_func_dir, i)

    if patient_func_path == None:
        print('Error: No {} images found for the specified patient. The pipeline cannot proceed. Please ensure that all filenames adhere to the BIDS standard. No NIFTI files with the extension \'_{}.nii.gz\' were detected. Exiting...'.format(DATATYPE_FILE_SUFFIX.upper(), DATATYPE_FILE_SUFFIX))
    else:
        preproc = buildWorkflow(patient_func_path, template_path, segment_path, outDir, args.subject_id[0], args.testmode, args.saveIntermediates)
        tic = time.time()
        preproc.run()
        toc = time.time()
        print('\nElapsed Time to Preprocess: {}s\n'.format(tic-toc))



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\tReceived Keyboard Interrupt, ending program.\n")
        sys.exit(2)