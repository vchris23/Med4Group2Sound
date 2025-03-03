class Track:
    def __init__(self, sound, label):
        self.sound = sound
        self.label:str = label
        self.source:str = ""
        self.train_or_test:str = ""
        self.original_track:str = ""
        self.features:list = []

#Tracks = [Track(1,"hello"), Track(2, "I am god"), Track("Me", "Behave")]

#labels = [track.label for track in Tracks]
#print(labels)