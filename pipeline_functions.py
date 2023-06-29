import nibabel as nib
import numpy as np

# Note: takes in the paths to the template and bold images and outputs the
# array of average intensity values for each brain region
def make_average_arr(bold_path, template_path):
    bold = nib.load(bold_path)
    template = nib.load(template_path)
    bold_array = bold.get_fdata()
    template_array = template.get_fdata() 
    _,_,_,timepoints = bold.shape
    structure_indices = template_array.max() + 1
    uniq_structure_indices = np.unique(template_array)
    avg_arr = np.zeros((int(timepoints),int(structure_indices)))
    for t in range(int(timepoints)):
        bold_time = bold_array[:,:,:,t]
        for s in range(int(structure_indices)):
            if s not in uniq_structure_indices:
                avg_arr[t,s] = 0. # to avoid finding average of missing indexes (i.e former thalami regions)
                continue
            template_indices = template_array == s
            matrix = bold_time[template_indices]
            res = np.average(matrix)
            avg_arr[t,s] = res
    avg_arr = np.nan_to_num(avg_arr) #redundant but do just incase
    return avg_arr


# Note: takes in the average intensity array from the previous function and  
# calculates the Pearson Correlation Coefficients to find # similarity between
# regions
def build_sim_arr(avg_arr):
    rows,columns = avg_arr.shape
    sim_matrix = np.full((columns,columns), np.nan)
    for c in range(columns):
        column_1 = avg_arr[:,c]
        for r in range(columns):
            column_2 = avg_arr[:, r]
            if np.any(np.isnan(column_1)) == True or np.any(np.isnan(column_2)) == True: 
                similarity = np.ma.corrcoef(np.ma.masked_invalid(column_1), np.ma.masked_invalid(column_2))
                print(f"Invalid value found in column {r} and {c}")[0,1]
            else:
                similarity = np.corrcoef(column_1, column_2)[0,1]
            sim_matrix[r, c] = similarity
    
    # ordinarily the diagonal would be 1, but we change it to 0 to be compliant
    # with adjacency networks (i.e no self connections)
    np.fill_diagonal(sim_matrix, 0.)
    return sim_matrix