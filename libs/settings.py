import pickle
import os

class Settings(object):
    def __init__(self):
        # Be default, the home will be in the same folder as labelImg
        home = os.path.expanduser("~")
        self.data = {}
        self.path = os.path.join(home, '.labelImgSettings.pkl')

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'rb') as f:
                    self.data = pickle.load(f)
                    return True
        except:
            print('Loading setting failed')
        return False
