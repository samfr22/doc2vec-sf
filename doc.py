class Doc:
    """
    Representation of a document for the model to use
    """
    
    def __init__(self, doc_id, text, labels = []):
        self.doc_id = doc_id
        self.text = text.split(" ")
        self.labels = [""] + labels

    def make_window(self, window_size):
        self.windows = []
        midpoint = int(window_size / 2)
        # Build a set of windows for the text
        for letters in range(len(self.text)):
            temp_window = []
            for x in range(letters - midpoint, letters + midpoint - 1):
                if x < 0 or x >= len(self.text):
                    # Edge of window 
                    temp_window.append("")
                else:
                    temp_window.append(self.text[x])
            # Current window built - add to the list
            self.windows.append(temp_window)