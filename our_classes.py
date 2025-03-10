class Track:
    def __init__(self, sound = None, label = None, source = None, original_track = None, name = None):
        self.sound = sound
        self.label:str = label
        self.source:str = source
        self.train_or_test:str = ""
        self.original_track:str = original_track
        self.name: str = name
        self.features:list = []

    def __str__(self):
        return f"name: {self.name}, label: {self.label}, source: {self.source}, original: {self.original_track}, features: {self.features}"

#Tracks = [Track(1,"hello"), Track(2, "I am god"), Track("Me", "Behave")]

#labels = [track.label for track in Tracks]
#print(labels)