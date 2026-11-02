# Data

Put skull-stripped axial slices here for fine-tuning. One folder per modality, then a cohort folder, a case folder, and the slice. The brain mask uses the same file name.

```
data/BrainGFM/
    T1/         <cohort>/<case>/slice_XXXX.nii.gz
    T1ce/       <cohort>/<case>/slice_XXXX.nii.gz
    T2/         <cohort>/<case>/slice_XXXX.nii.gz
    FLAIR/      <cohort>/<case>/slice_XXXX.nii.gz
    DWI/        <cohort>/<case>/slice_XXXX.nii.gz
    CT/         <cohort>/<case>/slice_XXXX.nii.gz
    Unknow/     <cohort>/<case>/slice_XXXX.nii.gz
    Lable/
        Brain_mask/<cohort>/<case>/slice_XXXX.nii.gz
        Lable_mask/<cohort>/<case>/slice_XXXX.nii.gz
        Merged_mask/<cohort>/<case>/slice_XXXX.nii.gz
```

`Lable_mask` is the task label. `Merged_mask` is optional; the loader builds it when the file is absent. Do not commit the images.
