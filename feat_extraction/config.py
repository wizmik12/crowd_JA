import pandas as pd

######  Data VLC
sicap_data = {"data_dir":"/data/datasets/SICAP/SICAPv2/images/",
	      "train_df": pd.read_excel("/data/datasets/SICAP/SICAPv2/partition/Validation/Val1/Train.xlsx"),
	      "val_df": pd.read_excel('/data/datasets/SICAP/SICAPv2/partition/Validation/Val1/Test.xlsx'),
	      "test_df": pd.read_excel('/data/datasets/SICAP/SICAPv2/partition/Test/Test.xlsx')}
	     
####### Data GRX

def append_ext(fn):
    return fn+".jpg"
    

train_df = pd.read_csv('/work/work_mik/crowd_JA/feat_extraction/labels_grx/train_split.csv')
train_df["Patch filename"]=train_df["Patch filename"].apply(append_ext)

val_df = pd.read_csv('/work/work_mik/crowd_JA/feat_extraction/labels_grx/val_split.csv')
val_df["Patch filename"]=val_df["Patch filename"].apply(append_ext)

test_df = pd.read_csv('/data/Prostata/BDs/Partitions/images_UGR/test.csv')
test_df["Patch filename"]=test_df["Patch filename"].apply(append_ext)

grx_data = {"data_dir": "/data/Prostata/BDs/Partitions/images_UGR/images/",
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df}

