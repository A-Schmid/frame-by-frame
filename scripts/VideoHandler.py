import config
import video_converter
from glob import glob
import numpy as np
import pandas as pd

class VideoHandler():

    def __init__(self, category, name, file_id=0):
        self.load_config()

        paths = sorted(glob(f'{self.result_path}/{category}/{name}_frame_*.csv'))

        self.data = pd.concat([pd.read_csv(f) for f in paths])
        self.set_file_id(file_id)

        print(len(self.data.index))
        print(self.data.head())

        #if os.path.exists(f'{result_path}/{category}/{name}_frame_{frame_index:03d}.csv'):

        #self.df = pd.DataFrame(columns=['frame', 'action', 'probability'])
        pass

    def set_file_id(self, file_id):
        self.file_id = file_id
        self.data['file'] = file_id

    def set_result_path(self, result_path):
        self.result_path = result_path

    def load_config(self):
        self.set_result_path(config.RESULT_PATH)

    #def add_video(self, path):
    #    pass

    # saves stored data in the desired format
    def save(self, file_format='csv'):
        pass

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

    def generate_matrix(self, method, file_id_1, frame_id_1, file_id_2, frame_id_2, categories=[]):
        # not sure what it's comparing - better ask the experts
        pass

    # returns the most important frame for specified file
    def get_mif(self, file_id, method):
        mif = 0
        return mif

    # trains the model
    def train(self):
        pass

    # gets segment with specified length around mif
    # can use different algorithms:
    #  * mif in the beginning/middle/end
    #  * maximize importance
    def get_segment(self, length, method):
        pass