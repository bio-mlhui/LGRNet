





# Data Preparation

## UFUV (Private): 
please email the second author for UFUV dataset if you want, I have no absolute power for UFUV

## VPS (Public)

### CVC/Kvasir/Mayo
We follow [PNS-Net](https://github.com/GewelsJI/PNS-Net) to download the CVC/Kvasir/Mayo dataset. The download link is same as [link](https://drive.google.com/file/d/1TyaRy4c4nHFDa3o2bOl4dP5Z7wes7HV2/view?usp=sharing)

Put MICCAI-VPS-dataset.zip in $DATASET_PATH, then run following script to change the directory structure:
```
cd $DATASET_PATH
unzip -qq MICCAI-VPS-dataset.zip

# cd LGRNet directory
# normalize the VPS data structure
python handle_vps.py
```

Now the structure should be like:
```
${DATASET_PATH}
    -- MICCAI-VPS-dataset
        -- Kvasir-SEG
            -- *
        -- VPS-TestSet
            -- CVC-ColonDB-300
                -- *
            -- CVC-ClinicDB-612-Valid
                -- *		
            -- CVC-ClinicDB-612-Test
                -- *
        -- VPS-TrainSet
            -- ASU-Mayo_Clinic
                -- Train
                    -- *
            -- CVC-ClinicDB-612	
                -- Train
                    -- *
            -- CVC-ColonDB-300
                -- Train
                    -- *
```
where * means the following structure:
```
-- Frame
    -- vid1
        -- img file
-- GT
    -- vid1
        -- mask file
```

### SUN-SEG

Please follow https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md to email the author for SUN-SEG.
Put part1, part2, annotation in $DATASET_PATH/SUN-SEG

```
# normalize the directory
unzip -qq $DATASET_PATH/SUN-SEG/sundatabase_positive_part1.zip -d $DATASET_PATH/SUN-SEG/SUN-Positive
unzip -qq $DATASET_PATH/SUN-SEG/sundatabase_positive_part2.zip -d $DATASET_PATH/SUN-SEG/SUN-Positive
tar -xf $DATASET_PATH/SUN-SEG/SUN-SEG-Annotation.tar -C $DATASET_PATH/SUN-SEG/

rm -rf $DATASET_PATH/SUN-SEG/SUN-SEG-Annotation/TestEasyDataset/Unseen/Frame
find $DATASET_PATH/SUN-SEG/SUN-SEG-Annotation -name "._.DS_Store" -type f -delete
python reorganize_sunseg.py

```
Now the structure should be like:
```
${DATASET_PATH}
    -- SUN-SEG
        -- SUN-SEG-Annotation
            -- TrainDataset
                -- *
            -- TestEasyDataset
                -- combine
                    -- *
            -- TestHardDataset
                -- combine
                    -- *
```



