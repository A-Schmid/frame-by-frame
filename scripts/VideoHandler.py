import config
import video_converter
from glob import glob
import os
import numpy as np
import pandas as pd
import subprocess

from time import perf_counter

# about 4 ms on lab server
def get_processed_videos(path=config.RESULT_PATH):
    categories = os.listdir(path)
    result = []

    #print(subprocess.Popen(['pwd'], stdout=subprocess.PIPE,).stdout)

    for category in categories:
        #files = paths = glob(f'{path}/{category}/*_frame_000.csv')
        files = [f for f in os.listdir(f'{path}/{category}') if '_frame_000' in f]

        #files = subprocess.Popen(f'ls {path}/{category}/*_frame_000.csv', shell=True, stdout=subprocess.PIPE,).stdout
        #files = subprocess.Popen(['ls'], stdout=subprocess.PIPE,).stdout

        names = []
        for f in files:
            #name = f.split('/')[-1].split('_frame')[0]
            name = str(f).split('_frame')[0]

            #names.append(name)

            result.append({'name' : name, 'category' : category})

            #if name not in names:
            #    names.append(name)
        #for name in names:
        #    result.append({'name' : name, 'category' : category})

    return result

#def handlers_to_df(handlers):
#    df = pd.DataFrame(columns=['name', 'file_id', 'frame_count', 'category', 'accuracy'])
#    for handler in handlers:
#        df = df.append(handler.to_dict(), ignore_index=True)
#    return df

class VideoHandler():

    counter = 0

    # might perform poorly with large data set
    handlers = dict()
    video_data = pd.DataFrame(columns=['name', 'file_id', 'frame_count', 'category', 'accuracy'])

    def get_categories():
        return sorted(list(VideoHandler.video_data['category'].unique()))

    # returns a list of accuracies of all videos in the given category
    # Todo: make param a list (example: get all sports-related things) 
    def get_accuracy_for_category(category):
        result = []
        for handler in VideoHandler.handlers.values():
            if handler.get_category() == category:
                result.append(handler.get_total_accuracy())
        return result

    # averages probabilites for each action for videos with the given category
    # this one is needed for softmax_rdm
    def get_probability_vector_for_category(category):
        result = []

        videos = VideoHandler.video_data[VideoHandler.video_data['category'] == category]['file_id']

        for video in videos.values:
            result.append(VideoHandler.handlers[video].get_probabilities())

        #for handler in VideoHandler.handlers.values():
        #    if handler.get_category() == category:
        #        result.append(handler.get_probabilities())

        df = pd.DataFrame(result)
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.mean()
        df['category'] = category

        return df.to_dict()


    def __init__(self, category, name):
        self.load_config()
        self.set_category(category)
        self.set_name(name)

        # get list of csv files for the video
        paths = sorted(glob(f'{self.result_path}/{category}/{name}_frame_*.csv'))
        self._frame_count = len(paths)

        # create data frame for the video by concatenating data for all frames
        # this step takes 100-150 ms on the lab server
        self.data = pd.concat([pd.read_csv(f, dtype={'file_id' : 'int16', 'action' : 'category'}) for f in paths])

        #file_id = name.split('_')[-1]
        #self.set_file_id(file_id)
        self.set_file_id(VideoHandler.counter)

        #print(len(self.data.index))
        #print(self.data.head())

        #if os.path.exists(f'{result_path}/{category}/{name}_frame_{frame_index:03d}.csv'):

        #self.df = pd.DataFrame(columns=['frame', 'action', 'probability'])

        VideoHandler.video_data = VideoHandler.video_data.append(self.to_dict(), ignore_index=True)
        VideoHandler.handlers[self.get_file_id()] = self
        VideoHandler.counter += 1

    def to_dict(self):
        result = dict()
        result['name'] = self.get_name()
        result['file_id'] = self.get_file_id()
        result['frame_count'] = self.get_frame_count()
        result['category'] = self.get_category()
        result['accuracy'] = self.get_total_accuracy()
        return result

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

    def get_file_id(self):
        return self.file_id

    def get_name(self):
        return self.name

    def get_probabilities(self):
        #df = self.data[self.data['action'] == self._category]
        #return list(df['probabilities'])
        result = dict()
        for category in self.data['action'].unique():
            result[category] = self.data[self.data['action'] == category]['probability'].mean()
        return result


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
