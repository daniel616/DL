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
    val=val[:50]
    test=val[:100]

    train.to_csv(loc+"DL_train_toy.csv")
    test.to_csv(loc+"DL_test_toy.csv")
    val.to_csv(loc+"DL_valid_toy.csv")

def make_csvs_range(loc=file_locs.csv_dir,range=(0.4,0.6)):
    df = pd.read_csv(file_locs.csv_dir + "DL_info.csv")
    df['z'] = df.Normalized_lesion_location.apply(lambda x: float(x.split(", ")[-1]))
    df=df[df.z>range[0]]
    df=df[df.z<range[1]]
    train, val, test = df[df.Train_Val_Test == 1], df[df.Train_Val_Test == 2], df[df.Train_Val_Test == 3]

#WQERQWER
    train.to_csv(loc + "DL_train_body.csv")
    test.to_csv(loc + "DL_test_body.csv")
    val.to_csv(loc + "DL_valid_body.csv")
#HIIISDF
if __name__ == "__main__":
    #make_csvs_range()
    make_csvs_toy()
