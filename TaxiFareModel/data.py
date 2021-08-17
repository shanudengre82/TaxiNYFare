import pandas as pd
from sklearn.model_selection import train_test_split

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"

def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    # if nrows<10000:
    #     df_toreturn = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)

    # else:
    #     df_toreturn = pd.read_csv(AWS_BUCKET_PATH, nrows=10000)
    #     parts = nrows//10000
    #     remainder = nrows%10000

    #     if parts == 1:
    #         df = pd.read_csv(AWS_BUCKET_PATH, skiprows=10000, nrows=remainder)
    #         df.columns = list(df_toreturn)
    #         df_toreturn = pd.concat([df_toreturn, df], ignore_index=True)

    #     else:
    #         for i in range(parts):
    #             df = get_data(skiprows=(i+1)*10000, nrows=10000)
    #             df.columns = list(df_toreturn)
    #             #train_10K = train_10K.append(df, ignore_index=True)
    #             df_toreturn = pd.concat([df_toreturn, df], ignore_index=True)
    #         df_toreturn = pd.concat([df_toreturn, get_data(skiprows=(i+1)*10000, remainder=10000)],
    #                                 ignore_index=True)

    df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


# def holdout(df):
#     X = df.drop(columns="fare_amount")
#     y = df.fare_amount

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#     return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = get_data()
