import pandas as pd
import numpy as np

data = "/mnt/d/Data/mangaki-data-challenge/"

def main():
    # w2v feature
    i1 = pd.read_csv(data+"features/item_w2v_32d.csv")
    u1 = pd.read_csv(data+"features/user_w2v_32d.csv")
    # w2v negative feature
    i2 = pd.read_csv(data+"features/item_w2vneg_10d.csv")
    u2 = pd.read_csv(data+"features/user_w2vneg_10d.csv")
    # lda feature
    i3 = pd.read_csv(data+"features/item_lda_20d.csv")
    u3 = pd.read_csv(data+"features/user_lda_20d.csv")
    # lda negative feature
    #i4 = pd.read_csv(data+"features/item_ldaneg_10d.csv")
    #u4 = pd.read_csv(data+"features/user_ldaneg_10d.csv")
    # lsi feature
    i5 = pd.read_csv(data+"features/item_lsi_20d.csv")
    u5 = pd.read_csv(data+"features/user_lsi_20d.csv")

    # merge with baseline feature file
    for i in [0, 1, 2, 3]:
        train = pd.read_csv(data+"baseline/train_{0}.csv".format(i))
        if i != 0:
            valid = pd.read_csv(data+"baseline/valid_{0}.csv".format(i))
        else:
            valid = pd.read_csv(data+"baseline/test_{0}.csv".format(i))

        train = train.merge(i2, on='work_id', how='left').\
                      merge(u2, on='user_id', how='left').\
                      merge(i1, on='work_id', how='left').\
                      merge(u1, on='user_id', how='left').\
                      merge(i3, on='work_id', how='left').\
                      merge(u3, on='user_id', how='left').\
                      merge(i5, on='work_id', how='left').\
                      merge(u5, on='user_id', how='left')
        valid = valid.merge(i2, on='work_id', how='left').\
                      merge(u2, on='user_id', how='left').\
                      merge(i1, on='work_id', how='left').\
                      merge(u1, on='user_id', how='left').\
                      merge(i3, on='work_id', how='left').\
                      merge(u3, on='user_id', how='left').\
                      merge(i5, on='work_id', how='left').\
                      merge(u5, on='user_id', how='left')

        train.to_csv(data+"latest/train_{0}.csv".format(i), index=False)
        if i != 0:
            valid.to_csv(data+"latest/valid_{0}.csv".format(i), index=False)
        else:
            valid.to_csv(data+"latest/test_{0}.csv".format(i), index=False)

if __name__=="__main__":
    main()