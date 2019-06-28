import file_locs
import pandas as pd

def make_csvs(loc=file_locs.csv_dir):
    df=pd.read_csv(file_locs.csv_dir+"DL_info.csv")
    train,val,test=df[df.Train_Val_Test==1],df[df.Train_Val_Test==2],df[df.Train_Val_Test==3]
    train.to_csv(loc+"DL_train.csv")
    test.to_csv(loc+"DL_test.csv")
    val.to_csv(loc+"DL_valid.csv")

def make_csvs_toy(loc=file_locs.csv_dir):
    df=pd.read_csv(file_locs.csv_dir+"DL_info.csv")
    train,val,test=df[df.Train_Val_Test==1],df[df.Train_Val_Test==2],df[df.Train_Val_Test==3]
    train=train[:100]
    val=val[:15]
    test=val[:15]

    train.to_csv(loc+"DL_train_toy.csv")
    test.to_csv(loc+"DL_test_toy.csv")
    val.to_csv(loc+"DL_valid_toy.csv")

if __name__ == "__main__":
    make_csvs_toy()
