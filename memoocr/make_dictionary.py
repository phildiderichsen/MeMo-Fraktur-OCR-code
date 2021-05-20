# Script for creating unigram and bigram dictionary
# to be used in symspell for spelling corection

# import libraries
import configparser
from collections import Counter
from datetime import datetime
import string, os, sys

# empty boxes for outputs
dictionary = Counter()
bigrams_dict = Counter()


def make_dic(metadir):
    # Check if symspell data already exists
    frequency_out = os.path.join(metadir, 'frequency_dict_da_sm.txt')
    bigram_out = os.path.join(metadir, 'bigrams_dict_da_sm.txt')
    # If so, exit script
    if os.path.exists(frequency_out) and os.path.exists(bigram_out):
        print("SymSpellPy data already exists!\n")

    # Otherwise, carry on
    else:
        # TODO: Common crawl data not available in meta dir ...
        # main data flow
        i = 0
        # for every entry in the common crawl data
        print("...creating latin unigram and bigram list...")
        common_crawl = os.path.join(metadir, 'da.txt')

        # Go through common crawl data and count one-word and two-word phrases
        for line in common_crawl:
            # remove punctuation
            content = line \
                .translate(line.maketrans('', '', string.punctuation)) \
                .lower() \
                .split()
            try:
                # lol
                if content[0] == "lorem" and content[1] == "ipsum":
                    pass
                else:
                    # strip whitespace
                    content = [c.strip() for c in content]
                    # create bigram list
                    bigrams = []
                    # for each word in content
                    for c_idx in range(len(content) - 1):
                        # exract word and word + 1
                        bigrams.append((content[c_idx], content[c_idx + 1]))
                    # updatefd
                    bigrams_dict.update(bigrams)
                    dictionary.update(content)
                    # print current value
                    if i % 10000 == 0 and i != 0:
                        print("Dictionary size:", len(dictionary))
                    i += 1
                    # set max dicitonary length
                    if len(dictionary) > 10000000:
                        break
            except IndexError:
                pass

        print("...sorting outputs...")
        # Max lengths
        word_count = 1500000
        bigram_count = word_count * 4

        # create unigram list
        words = sorted(dictionary.items(), key=lambda item: item[1])
        words = words[::-1]
        words = words[:1500000]

        # create bigram list
        bigrams = sorted(bigrams_dict.items(), key=lambda item: item[1])
        bigrams = bigrams[::-1]
        bigrams = bigrams[:bigram_count]

        print("...saving and closing...")
        # write everything to files
        print("saving unigrams")
        with open(frequency_out, "w+") as w:
            for elem in words:
                w.write("{} {}\n".format(elem[0], elem[1]))

        print("saving bigrams")
        with open(bigram_out, "w+") as w:
            for elem in bigrams:
                w.write("{} {} {}\n".format(elem[0][0], elem[0][1], elem[1]))

        print("...done!")


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini'))

    mode = 'test'
    conf = config['DEFAULT']
    metadir = conf['metadir']
    make_dic(metadir)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == "__main__":
    main()