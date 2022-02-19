class ActionData():

    def __init__(self):
        pass

    def add_video(self, path):
        pass

    # saves stored data in the desired format
    def save(self, file_format='csv'):
        pass

    # combine redundant actions to new_action (e.g. singing_child and singing_adult to singing)
    # default method: average
    def combine_actions(self, actions, new_action):
        pass

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
