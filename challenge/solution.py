import os
import itertools
import pandas as pd

def get_data(timestamp, input_file):
    timestamp = pd.Timestamp(timestamp)
    data = pd.read_csv(input_file)
    data['time'] = data['time'].apply(
        lambda dt: dt[:-5] + '00:00'
    )
    data['time'] = pd.to_datetime(data['time'])
    data = data.loc[data.time <= timestamp]

    return data


def fill_nan(data):
    data = data.groupby(
        ['device', 'time']
    )['device_activated'].agg(
        'max'
    ).reset_index()

    data = data.set_index(
        ['time', 'device']
    ).unstack(
        fill_value=0
    ).asfreq(
        'H', fill_value=0
    ).stack().sort_index(
        level=1
    ).reset_index()

    return data


def extract_feature(data):
    data['day_name'] = data.time.dt.day_name()
    data['day'] = data.time.dt.day
    data['week'] = data.time.dt.isocalendar().week
    data['hour'] = data.time.dt.hour

    data = data[[
        'device',
        'day_name',
        'hour',
        'device_activated',
        'time'
    ]]
    return data


def data_pipeline(timestamp, input_file):
    data = get_data(timestamp, input_file)
    data = fill_nan(data)
    data = extract_feature(data)

    return data


def get_pred_hash(timestamp, input_file, output_file):
    data = data_pipeline(timestamp, input_file)
    timestamp = pd.Timestamp(timestamp)
    train_data = data[data.time <= timestamp]

    pred_hash = train_data.groupby(
        ['device', 'day_name', 'hour']
    ).agg(
        device_activated=('device_activated', 'max')
    ).reset_index()

    return pred_hash


def generate_prediction_holder(timestamp):
    timestamp = pd.Timestamp(timestamp)
    split_point = timestamp.round(freq='T')
    next_24_hours = pd.date_range(split_point, periods=24, freq='H').ceil('H')
    device_names = ['device_' + str(i) for i in range(1,8)]

    xproduct = list(itertools.product(next_24_hours, device_names))
    prediction_holder = pd.DataFrame(xproduct, columns=['time', 'device'])

    prediction_holder['day_name'] = prediction_holder.time.dt.day_name()
    prediction_holder['hour'] = prediction_holder.time.dt.hour
    columns = [
        'time',
        'device',
        'day_name',
        'hour'
    ]

    prediction_holder = prediction_holder[columns]

    return prediction_holder


def make_pred(timestamp, input_file, output_file):
    pred_hash = get_pred_hash(timestamp, input_file, output_file)
    prediction_holder = generate_prediction_holder(timestamp)
    pred = prediction_holder.merge(pred_hash, how='left', on=['device', 'day_name', 'hour'])
    pred = pred[[
        'time',
        'device',
        'device_activated'
    ]]
    pred = pred.fillna(0)
    pred.loc[pred['device'] == 'device_7', 'device_activated'] = 0 # we set like this because room 7 is highly rarely used.

    return pred



if __name__ == '__main__':
    timestamp = '2016-08-31 23:59:59'
    input_file = 'data\device_activations.csv'
    output_file = 'data\myresult.csv'

    pred = make_pred(timestamp, input_file, output_file)
    print(pred)
