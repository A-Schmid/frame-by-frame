import config
from glob import glob
import os
import numpy as np
import pandas as pd

def get_processed_videos(path=config.RESULT_PATH):
    categories = os.listdir(path)
    result = []

    for category in categories:
        files = [f for f in os.listdir(f'{path}/{category}')]
        names = []

        for f in files:
            name = str(f).split('.')[0]
            result.append({'name' : name, 'category' : category})

    return result

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

        df = pd.DataFrame(result)
        df = df.reindex(sorted(df.columns), axis=1)
        df = df.mean()
        df['category'] = category

        return df.to_dict()

    def from_path(path):
        # Todo: can fail if file in the same directory is specified
        # solution: use os.getcwd() in this case
        path_hierarchy = path.split('/')
        category = path_hierarchy[-2]
        name = path_hierarchy[-1].split('.')[0]
        return VideoHandler(category, name)

    def __init__(self, category, name):
        self.load_config()
        self.set_category(category)
        self.set_name(name)

        # read video data
        self.data = pd.read_csv(f'{self.result_path}/{category}/{name}.csv', dtype={'frame' : 'int8', 'action' : 'category'})
        self._frame_count = self.data['frame'].max()

        # file ID is currently static and only based on the order VideoHandler objects are created
        # Todo: make it an argument and use IDs from file list
        self.set_file_id(VideoHandler.counter)

        # set static variables
        VideoHandler.video_data = VideoHandler.video_data.append(self.to_dict(), ignore_index=True)
        VideoHandler.handlers[self.get_file_id()] = self
        VideoHandler.counter += 1

    # returns summary of the VideoHandler object as a dict
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

    def get_video_path(self):
        # Todo
        pass

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

    def get_accuracy_curve(self):
        df = self.data[self.data['action'] == self._category]
        accuracy = df[['frame', 'probability']]
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
    def get_mif(self):
        df = self.data[self.data['action'] == self._category]
        mif_row = df.loc[df['probability'] == df['probability'].max()]
        mif = mif_row['frame'].values[0]
        return mif

    def _get_max_probability_segment(self, length):
        df = self.data[self.data['action'] == self._category]

        max_sum = 0
        max_sum_index = 0

        for i in range(0, self.get_frame_count() - length - 1):
            start = i
            end = i + length
            df_slice = df[(df['frame'] >= start) & (df['frame'] < end)]
            probability_sum = df_slice['probability'].sum()
            if probability_sum > max_sum:
                max_sum = probability_sum
                max_sum_index = i

        return (max_sum_index, max_sum_index + length)

    def _get_max_minimum_segment(self, length):
        df = self.data[self.data['action'] == self._category]

        max_minimum = 0
        max_minimum_index = 0

        for i in range(0, self.get_frame_count() - length - 1):
            start = i
            end = i + length
            df_slice = df[(df['frame'] >= start) & (df['frame'] < end)]
            probability_min = df_slice['probability'].min()
            if probability_min > max_minimum:
                max_minimum = probability_min
                max_minimum_index = i

        return (max_minimum_index, max_minimum_index + length)

    # maybe move to util?
    def _shift_segment(self, segment):
        (start, end) = segment
        underflow = 0 - start
        overflow = (end - 1) - self.get_frame_count()

        if underflow > 0:
            end += underflow
            start = 0
        elif overflow > 0:
            end = self.get_frame_count() - 1
            start -= overflow

        return (start, end)

    # gets segment with specified length around mif
    # can use different algorithms:
    #  * mif in the beginning/middle/end
    #  * maximize importance
    #  * maximize minimum importance
    def get_segment(self, length, method='mif_center'):
        if length > self.get_frame_count():
            # Todo: what to we do?
            return None

        mif = self.get_mif()
        start = 0
        end = self.get_frame_count()

        if method == 'mif_beginning':
            start = mif
            end = mif + length
        elif method == 'mif_center':
            start = int(mif - (length / 2))
            end = start + length
        elif method == 'mif_end':
            end = mif
            start = mif - length
        elif method == 'max_probability':
            start, end = self._get_max_probability_segment(length)
        elif method == 'max_minimum':
            start, end = self._get_max_minimum_segment(length)

        segment = (start, end)
        segment = self._shift_segment(segment)

        return segment
