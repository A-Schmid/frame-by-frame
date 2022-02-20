import config
import video_converter
from glob import glob
import numpy as np
import pandas as pd

class VideoHandler():

    def __init__(self, category, name, file_id=0):
        self.load_config()
        self.set_category(category)
        self.set_name(name)

        # get list of csv files for the video
        paths = sorted(glob(f'{self.result_path}/{category}/{name}_frame_*.csv'))
        self._frame_count = len(paths)

        # create data frame for the video by concatenating data for all frames
        self.data = pd.concat([pd.read_csv(f) for f in paths])
        self.set_file_id(file_id)

        print(len(self.data.index))
        print(self.data.head())

        #if os.path.exists(f'{result_path}/{category}/{name}_frame_{frame_index:03d}.csv'):

        #self.df = pd.DataFrame(columns=['frame', 'action', 'probability'])
        pass

    def set_category(self, category):
        self._category = category

    def set_name(self, name):
        self.name = name

    def set_file_id(self, file_id):
        self.file_id = file_id
        self.data['file'] = file_id

    def set_result_path(self, result_path):
        self.result_path = result_path

    def get_frame_count(self):
        return self._frame_count

    def get_category(self):
        return self._category

    #def get_accuracies(self):
    #    df = self.data[self.data['action'] == self._category]
    #    return list(df['probabilities'])

    # get accuracy for the whole video
    # accuracies for all frames are averaged with the passed method (default: mean)
    # the passed average function has to take a pandas series as an argument
    def get_total_accuracy(self, average=np.mean):
        df = self.data[self.data['action'] == self._category]
        accuracy = average(df['probability'])
        return accuracy

    def load_config(self):
        self.set_result_path(config.RESULT_PATH)

    #def add_video(self, path):
    #    pass

    ## saves stored data in the desired format
    #def save(self, file_format='csv'):
    #    pass

    # combine redundant actions to new_action (e.g. singing_child and singing_adult to singing)
    # default method: average
    def combine_actions(self, actions, new_action):
        for frame in self.data['frame'].unique():
            values = []
            for action in actions:
                df = self.data[self.data['frame'] == frame]
                values.append(df[df['action'] == action]['probability'].values[0])
            self.data = self.data.append({'frame' : frame, 'action' : new_action, 'probability' : np.mean(values), 'file' : self.file_id}, ignore_index=True)
        self.data = self.data[self.data.action.isin(actions) == False]

    ## might be better somewhere else
    #def generate_matrix(self, method, file_id_1, frame_id_1, file_id_2, frame_id_2, categories=[]):
    #    # not sure what it's comparing - better ask the experts
    #    pass

    # returns slice of data set for one frame (equal to one CSV file)
    # Todo: currently unused
    def get_frame_data(self, frame):
        # todo exception handling
        #  * frame not in number of frames
        if frame < 0 or frame >= self._frame_count:
            return None

        return self.data[self.data['frame'] == frame]

    # returns the most important frame for specified file
    # Todo: split up into multiple private functions as more methods are added
    def get_mif(self, method=''):
        df = self.data[self.data['action'] == self._category]
        mif_row = df.loc[df['probability'] == df['probability'].max()]
        mif = mif_row['frame'].values[0]
        return mif

    # trains the model
    #def train(self):
    #    pass

    # gets segment with specified length around mif
    # can use different algorithms:
    #  * mif in the beginning/middle/end
    #  * maximize importance
    def get_segment(self, length, method):
        pass
