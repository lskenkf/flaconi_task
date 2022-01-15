import sys
import itertools
import pandas as pd


"""
Hi, first of all I want to think you for reviewing my solution. My answers are formed by two parts. In the first one I 
will explain my idea and implementation of the model. In the second part, I will address potential points which could 
improve the model performance.

1.

Now, I want to talk briefly about the idea and process behind the model formulation.
Since we want to forecast the occupancy of each room given a split point, whose next hour will be the first hour we have to 
predict, my first thought is to use the historical data to predict the next 24 hours' occupancy of each meeting room using 
regular ml approach, i.e., use model to predict.

I abondan it because firstly, we need to do autoregression, i.e., from starting point t we want to predict until 
t + 24, but this uses previous prediction as input for next prediction, that is to say we will add errors into the whole 
model. Secondly, in order to let the model work, we have to impute the time when the room is not occupied,i.e., add a negative 
target to let the model to learn. If we do so, we will face a very imbalanced dataset, it will also make the model perform 
very week. Although, we can try some rebalancing technics, but these do not solve the problem completely.

My second attempt is to use device, day, hour info of each timestamp, so that we can have a mapping like 
map(device_name, day_name,hour) -> occupancy_next_hour, I did this with a regression model from automl packages, but it 
performed very weakly. I think the reason is due to the imbalanced data issue.

Still I want the mapping like above. I think maybe, the pattern of previous data can be used like a hash mapping for the future 
meeting room occupancy. At last, what I did is to aggregate the historical data of meeting occupancy of each device, so 
if we want to predict for friday from 0 to 23 for device_1, then we can extract the info from this hash table. That is to say, the pattern 
of futuer room occupancy shall follow the previous room occupancy pattern.

After some model evaluation, I think it is stable and is a much more reliable solution.
At the end, I implemented this one with a simple class whose explanation can be found below.

2. 
In the model evaluation process, I checked the performance for each timestamp, and found out that we always have false positives,
i.e., we predict that the room will be occupied but it is not. If I have more time, I think I can do some more data analysis to figure 
out the reason of the false positive. Afterwards. I can upgrade the model with some proper approaches.
  

"""



class frequentist_model:
    """
        The frequentist_model contains three parameters

        Attributes
        ----------
        timestamp: str
            The starting time from where we need to predict the next 24 hours' meeting room occupancy.
        input_file: str
            Where we can get the training set
        output_file: str
            Where we shall store the prediction.


        Methods
        -------
        get_data()
            This is the first step of the data pipeline where we load the csv file as a pandas.dataFrame for next processing.

        fill_nan(data)
            This is the second part of the data pipeline. It returns the training data filled with missing hour infos,
            because we do not have any info about when the sensor is not activated.
            If there is no record for that hour, we just fill 0 to it alongeside with right time.

        extract_feature(data)
            The is the last part of the data pipeline where we extract week, day and hour info from time column.

        data_pipeline()
            The wrap up of the above three methods.

        get_pred_hash()
            We aggregate historical occupacy for each device into one week.

        generate_prediction_holder()
            The is the wrap up of the solution.py file. We will use the returned table to join with the pred_hash
            table from get_pred_hash().

        make_pred()
            We join the return from generate_prediction_holder() and the return from get_pred_hash().
            Afterwars we write the result to self.output_file. Moreover, we return the same pandas.dataFrame for model evaluation.
    """

    def __init__(self,timestamp, input_file, output_file):
        self.timestamp = timestamp
        self.input_file = input_file
        self.output_file = output_file

    def get_data(self):
        """
        Description:
            This is the first step of the data pipeline where we load the csv file as a pandas.dataFrame for next processing.

        Returns:
            pandas.dataFrame: The pandas.dataFrame contains only the info about time, device and device_activated when
            the sensor is activated.
        """

        timestamp = pd.Timestamp(self.timestamp)
        data = pd.read_csv(self.input_file)
        data['time'] = data['time'].apply(
            lambda dt: dt[:-5] + '00:00'
        )
        data['time'] = pd.to_datetime(data['time'])
        data = data.loc[data.time <= timestamp]

        return data


    def fill_nan(self,data):
        """
        Description:
            This is the second part of the data pipeline. It returns the training data filled with missing hour infos,
            because we do not have any info about when the sensor is not activated.
            If there is no record for that hour, we just fill 0 to it alongeside with right time.

        Parameters:
            data (pandas.dataFrame): The output we get from get_data(timestamp, input_file).

        Returns:
            pandas.dataFrame: The pandas.dataFrame contains all the info about time, device and device_activated for each hour.
        """

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


    def extract_feature(self,data):
        '''
        Description:
            The is the last part of the data pipeline where we extract week, day and hour info from time column.

        Parameters:
            data (pandas.dataFrame): The output we get from fill_nan(data).

        Returns:
            pandas.dataFrame: The data with all the info we need.
        '''

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


    def data_pipeline(self):
        '''
        Description:
            The wrap up of the above three methods.

        Returns:
            pandas.dataFrame: The data with all the info we need.
        '''

        data = self.get_data()
        data = self.fill_nan(data)
        data = self.extract_feature(data)

        return data


    def get_pred_hash(self):
        '''
        Description:
            We aggregate historical occupacy for each device into one week.

        Returns:
            pandas.dataFrame: The data contains all the historical occupacy for one week.
        '''

        data = self.data_pipeline()
        timestamp = pd.Timestamp(self.timestamp)
        train_data = data[data.time <= timestamp]

        pred_hash = train_data.groupby(
            ['device', 'day_name', 'hour']
        ).agg(
            device_activated=('device_activated', 'max')
        ).reset_index()

        return pred_hash


    def generate_prediction_holder(self):
        '''
        Description:
            The is the wrap up of the solution.py file. We will use the returned table to join with the pred_hash table from
            get_pred_hash().

        Returns:
            pandas.dataFrame: Returns a place holder of occupancy prediction for each device at the next 24 hours.
        '''

        timestamp = pd.Timestamp(self.timestamp)
        split_point = timestamp.round(freq='T')
        next_24_hours = pd.date_range(split_point, periods=24, freq='H').ceil('H')
        # We consider the senario that a device will appear for the first time at the next 24 hours.
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


    def make_pred(self):
        '''
        Description:
            We join the return from generate_prediction_holder() and the return from get_pred_hash().
            Afterwars we write the result to self.output_file. Moreover, we return the same pandas.dataFrame for model evaluation.

        Returns:
            pandas.dataFrame:
        '''

        pred_hash = self.get_pred_hash()
        prediction_holder = self.generate_prediction_holder()
        pred = prediction_holder.merge(pred_hash, how='left', on=['device', 'day_name', 'hour'])
        pred = pred[[
            'time',
            'device',
            'device_activated'
        ]]
        pred = pred.fillna(0) # if a device/meeting room appears for the first time in test set, than we set its occupancy as 0.
        pred.loc[pred['device'] == 'device_7', 'device_activated'] = 0 # we set like this because room 7 is highly rarely used.
        pred.to_csv(output_file)

        return pred


if __name__ == '__main__':

    timestamp, input_file, output_file = sys.argv[1:]
    frequentist_model = frequentist_model(timestamp, input_file, output_file)
    pred = frequentist_model.make_pred()

    #solution.py "2016-08-29 00:00:00" data\device_activations.csv myresult.csv