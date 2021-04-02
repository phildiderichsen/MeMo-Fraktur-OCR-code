import os
import re


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """

    def atoi(_text):
        return int(_text) if _text.isdigit() else _text

    return [atoi(c) for c in re.split(r'(\d+)', text)]


def sorted_listdir(directory):
    """Version of listdir where the files are sorted in human order."""
    dirlist = os.listdir(directory)
    dirlist.sort(key=natural_keys)
    return dirlist
