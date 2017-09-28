import pandas as pd
import numpy as np

data = "/mnt/d/Data/mangaki-data-challenge/"

def record_basic_w2v_lda_lsi():
    # w2v feature
    i1 = pd.read_csv(data+"features/item_w2v_shuffled_32d.csv")
    u1 = pd.read_csv(data+"features/user_w2v_shuffled_32d.csv")
    # w2v negative feature
    i2 = pd.read_csv(data+"features/item_w2vneg_shuffled_10d.csv")
    u2 = pd.read_csv(data+"features/user_w2vneg_shuffled_10d.csv")
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

def wish_lsi_w2v():
    # w2v feature
    i1 = pd.read_csv(data+"features/itemwish_w2v_shuffled_20d.csv")
    u1 = pd.read_csv(data+"features/userwish_w2v_shuffled_20d.csv")
    i2 = pd.read_csv(data+"features/itemwish_w2vneg_shuffled_10d.csv")
    u2 = pd.read_csv(data+"features/userwish_w2vneg_shuffled_10d.csv")
    # lsi feature
    i3 = pd.read_csv(data+"features/itemwish_lsi_20d.csv")
    u3 = pd.read_csv(data+"features/userwish_lsi_20d.csv")

    train = pd.read_csv(data+"baseline/train_0.csv", usecols=["user_id","work_id","rating"])
    test = pd.read_csv(data+"baseline/test_0.csv", usecols=["user_id","work_id"])

    train = train.merge(i2, on='work_id', how='left').\
                  merge(u2, on='user_id', how='left').\
                  merge(i1, on='work_id', how='left').\
                  merge(u1, on='user_id', how='left').\
                  merge(i3, on='work_id', how='left').\
                  merge(u3, on='user_id', how='left')
    test = test.merge(i2, on='work_id', how='left').\
                  merge(u2, on='user_id', how='left').\
                  merge(i1, on='work_id', how='left').\
                  merge(u1, on='user_id', how='left').\
                  merge(i3, on='work_id', how='left').\
                  merge(u3, on='user_id', how='left')
    train.to_csv(data+"latest/train_0.csv", index=False)
    test.to_csv(data+"latest/test_0.csv", index=False)

    # merge with baseline feature file
    for i in [1, 2, 3]:
        # w2v feature
        i1 = pd.read_csv(data+"features/{0}/itemwish_w2v_shuffled_20d.csv".format(i))
        u1 = pd.read_csv(data+"features/{0}/userwish_w2v_shuffled_20d.csv".format(i))
        i2 = pd.read_csv(data+"features/{0}/itemwish_w2vneg_shuffled_10d.csv".format(i))
        u2 = pd.read_csv(data+"features/{0}/userwish_w2vneg_shuffled_10d.csv".format(i))
        # lsi feature
        i3 = pd.read_csv(data+"features/{0}/itemwish_lsi_20d.csv".format(i))
        u3 = pd.read_csv(data+"features/{0}/userwish_lsi_20d.csv".format(i))
        train = pd.read_csv(data+"baseline/train_{0}.csv".format(i), usecols=["user_id","work_id","rating"])
        valid = pd.read_csv(data+"baseline/valid_{0}.csv".format(i), usecols=["user_id","work_id","rating"])

        train = train.merge(i2, on='work_id', how='left').\
                    merge(u2, on='user_id', how='left').\
                    merge(i1, on='work_id', how='left').\
                    merge(u1, on='user_id', how='left').\
                    merge(i3, on='work_id', how='left').\
                    merge(u3, on='user_id', how='left')
        valid = valid.merge(i2, on='work_id', how='left').\
                    merge(u2, on='user_id', how='left').\
                    merge(i1, on='work_id', how='left').\
                    merge(u1, on='user_id', how='left').\
                    merge(i3, on='work_id', how='left').\
                    merge(u3, on='user_id', how='left')

        train.to_csv(data+"latest/train_{0}.csv".format(i), index=False)
        valid.to_csv(data+"latest/valid_{0}.csv".format(i), index=False)

def wish_basic_lsi_w2v():
    # basic feature
    i0 = pd.read_csv(data+"features/wish_itembasic.csv")
    u0 = pd.read_csv(data+"features/wish_userbasic.csv")
    # w2v feature
    i1 = pd.read_csv(data+"features/itemwish_w2v_shuffled_20d.csv")
    u1 = pd.read_csv(data+"features/userwish_w2v_shuffled_20d.csv")
    i2 = pd.read_csv(data+"features/itemwish_w2vneg_shuffled_10d.csv")
    u2 = pd.read_csv(data+"features/userwish_w2vneg_shuffled_10d.csv")
    # lsi feature
    i3 = pd.read_csv(data+"features/itemwish_lsi_20d.csv")
    u3 = pd.read_csv(data+"features/userwish_lsi_20d.csv")

    train = pd.read_csv(data+"baseline/train_0.csv", usecols=["user_id","work_id","rating"])
    test = pd.read_csv(data+"baseline/test_0.csv", usecols=["user_id","work_id"])

    train = train.merge(i0, on='work_id', how='left').\
                  merge(u0, on='user_id', how='left').\
                  merge(i2, on='work_id', how='left').\
                  merge(u2, on='user_id', how='left').\
                  merge(i1, on='work_id', how='left').\
                  merge(u1, on='user_id', how='left').\
                  merge(i3, on='work_id', how='left').\
                  merge(u3, on='user_id', how='left')
    test = test.merge(i0, on='work_id', how='left').\
                  merge(u0, on='user_id', how='left').\
                  merge(i2, on='work_id', how='left').\
                  merge(u2, on='user_id', how='left').\
                  merge(i1, on='work_id', how='left').\
                  merge(u1, on='user_id', how='left').\
                  merge(i3, on='work_id', how='left').\
                  merge(u3, on='user_id', how='left')
    train.to_csv(data+"latest/train_0.csv", index=False)
    test.to_csv(data+"latest/test_0.csv", index=False)

    # merge with baseline feature file
    for i in [1, 2, 3]:
        # basic feature
        i0 = pd.read_csv(data+"features/{0}/wish_itembasic.csv".format(i))
        u0 = pd.read_csv(data+"features/{0}/wish_userbasic.csv".format(i))
        # w2v feature
        i1 = pd.read_csv(data+"features/{0}/itemwish_w2v_shuffled_20d.csv".format(i))
        u1 = pd.read_csv(data+"features/{0}/userwish_w2v_shuffled_20d.csv".format(i))
        i2 = pd.read_csv(data+"features/{0}/itemwish_w2vneg_shuffled_10d.csv".format(i))
        u2 = pd.read_csv(data+"features/{0}/userwish_w2vneg_shuffled_10d.csv".format(i))
        # lsi feature
        i3 = pd.read_csv(data+"features/{0}/itemwish_lsi_20d.csv".format(i))
        u3 = pd.read_csv(data+"features/{0}/userwish_lsi_20d.csv".format(i))
        train = pd.read_csv(data+"baseline/train_{0}.csv".format(i), usecols=["user_id","work_id","rating"])
        valid = pd.read_csv(data+"baseline/valid_{0}.csv".format(i), usecols=["user_id","work_id","rating"])

        train = train.merge(i0, on='work_id', how='left').\
                    merge(u0, on='user_id', how='left').\
                    merge(i2, on='work_id', how='left').\
                    merge(u2, on='user_id', how='left').\
                    merge(i1, on='work_id', how='left').\
                    merge(u1, on='user_id', how='left').\
                    merge(i3, on='work_id', how='left').\
                    merge(u3, on='user_id', how='left')
        valid = valid.merge(i0, on='work_id', how='left').\
                    merge(u0, on='user_id', how='left').\
                    merge(i2, on='work_id', how='left').\
                    merge(u2, on='user_id', how='left').\
                    merge(i1, on='work_id', how='left').\
                    merge(u1, on='user_id', how='left').\
                    merge(i3, on='work_id', how='left').\
                    merge(u3, on='user_id', how='left')

        train.to_csv(data+"latest/train_{0}.csv".format(i), index=False)
        valid.to_csv(data+"latest/valid_{0}.csv".format(i), index=False)

if __name__=="__main__":
    wish_basic_lsi_w2v()