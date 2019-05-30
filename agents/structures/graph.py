class DictLen(object):
    def __init__(self):
        self.dict = {}
        self.lg = 0

    def append(self, element):
        if element not in dict:
            self.dict.update({element: self.lg})
            self.lg += 1
