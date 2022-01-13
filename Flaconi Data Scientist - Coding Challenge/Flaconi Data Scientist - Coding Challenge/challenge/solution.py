import pandas as pd


def get_train_data(timestamp, input_file):
    timestamp = pd.Timestamp(timestamp)
    timestamp_next_hour = timestamp.round(freq='H')
    data = pd.read_csv(input_file)
    data['time'] = pd.to_datetime(data['time'])
    data_time_max = data.time.max()

    if data_time_max > timestamp:
        print('the max timestamp of input file is ahead of the input timestamp, but we have make it right')
        data = data.loc[data.time <= timestamp]
    else:
        pass

    return data, timestamp_next_hour


def extract_date_info(data):
    data['day_name'] = data.time.dt.day_name()
    data['day'] = data.time.dt.day
    data['week'] = data.time.dt.week
    data['hour'] = data.time.dt.hour

    return data

if __name__ == '__main__':
    timestamp = '2016-08-30 00:59:59'
    input_file = 'data\device_activations.csv'
    data, timestamp_next_hour = get_train_data(timestamp, input_file)
    data = extract_date_info(data)
    print(str(data))
    