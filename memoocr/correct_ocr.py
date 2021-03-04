# Spelling correction using symspell from Oliver
import configparser
import os
from datetime import datetime
from symspellpy import SymSpell, Verbosity
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def spell_corrected(term, sym_spell, word_split):
    suggestions = sym_spell.lookup_compound(term,
                                            max_edit_distance=2,
                                            ignore_non_words=True,
                                            split_phrase_by_space=True,
                                            ignore_term_with_digits=False,
                                            transfer_casing=True)
    # Create lists to compare with/without punctuation
    # PD: This is brittle since a suggestion may well be a conflation of two 'word_split' items.
    # This means the chk_list may become shorter than the in_list, yielding an index error.
    # This is a segmentation problem right here.
    # => Fixing it by only doing the SymSpell correction when in_list and chk_list are equally long.
    corrected = suggestions[0].term
    in_list = word_split.findall(term)
    chk_list = word_split.findall(corrected)
    # To keep punctuation we take the original phrase and do word by word replacement
    # PD out_term = []
    # out_term = term
    out_term = []
    word_count = 0
    if len(in_list) == len(chk_list):
        for word in in_list:
            if len(word) == 1:
                out_term = term.replace(word, word)
            else:
                out_term = term.replace(word, chk_list[word_count])
            word_count += 1
            print('Out_term iteration:', out_term)
        print('Out_term:', out_term)
    return out_term


def correct_ocr(conf):
    print("Initialize SymSpell")
    # Initalise SymSpell
    sym_spell = SymSpell()
    dictionary_path = os.path.join(conf["metadir"], "frequency_dict_da_sm.txt")
    bigram_path = os.path.join(conf["metadir"], "bigrams_dict_da_sm.txt")
    sym_spell.load_dictionary(dictionary_path, 0, 1)
    sym_spell.load_bigram_dictionary(bigram_path,
                                     term_index=0,
                                     count_index=2)

    # Regex pattern1 - remove number from words like '55This'
    num_remove = re.compile(r"\d*([^\d\W]+)\d*")
    # Regex pattern2 - pad punctuation with whitespace
    punct_pad = re.compile('([.,:;„"»«\'!?()])')
    # Regex pattern3 - strip addtional whitespace
    whitespace = re.compile(r'\s{2,}')
    # Regex pattern1 Word split function, used to keep punctuation later
    word_split = re.compile(r"[^\W]+", re.U)

    # Sort novels, just because
    uncorrected_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    sorted_novels = os.listdir(uncorrected_dir)

    # For each novel
    for novel in sorted_novels:

        print(f"Working on {novel}")
        # For each page in that novel
        novel_pages = os.path.join(uncorrected_dir, novel)
        # Sort the pages, so that they're appended in the correct order
        sorted_pages = sorted(os.listdir(novel_pages))
        sorted_pages.sort(key=natural_keys)

        # For each page in the novel
        for page in sorted_pages:
            # Read the text in and save in a single list
            text = []
            with open(os.path.join(uncorrected_dir, novel, page), "r") as f:
                tmp = f.read()
                text.append(tmp)

            # Pad punctuation with whitespace, so they count as tokens
            text = re.sub(punct_pad, r' \1 ', ''.join(text))
            text = re.sub(whitespace, ' ', text)

            # list for sym_spell output
            output = []
            # Flatten lists into single string
            for word in text.split():
                # Remove obvious noise
                if word in ["*", "ð", "—", "——", "———", "—————"]:
                    pass
                # Keep informative punctuation
                elif word in ["—", ",", ".", ":", ";", "-", "?", "!", "'", '"']:
                    output.append(word)
                # If not noise of useful punctuation, use sym_spell to correct
                else:
                    option = sym_spell.lookup(word,
                                              Verbosity.TOP,
                                              max_edit_distance=2,
                                              transfer_casing=True)
                    if option:
                        suggestion = option[0]._term
                    else:
                        suggestion = word
                    # Append everything to output list
                    output.append(suggestion)

            joined = re.sub(punct_pad, r' \1 ', ' '.join(output))
            joined = re.sub(whitespace, ' ', joined)
            print('Joined:', joined)
            # Join corrected unigrams; check for bigrams
            suggested = spell_corrected(joined, sym_spell, word_split)
            # Join on punctuation again - replace with Regex
            suggested = ''.join(suggested).replace(" . ", ". ") \
                .replace(" , ", ", ") \
                .replace(" ; ", "; ") \
                .replace(" : ", ": ") \
                .replace(" ? ", "? ") \
                .replace(" ! ", "! ") \
                .replace(" “ ", " “ ") \
                .replace(" „ ", "„") \
                .replace(" “„ ", "“ „")

            # get filename by itself with no extension - for later
            name = ''.join(novel.split(".")[:1]) \
                .replace("uncorrected", "corrected")

            # Create output folder if not exists
            outfolder = os.path.join(conf['intermediatedir'], '3-corrected', novel)
            try:
                os.makedirs(outfolder)
            except FileExistsError:
                pass

            # Create outpath - merge all results into single text
            outpath = os.path.join(outfolder, f"{name}_corrected.txt")
            with open(outpath, "a") as f:
                # for result in suggested:
                #    f.writelines(f"{result}\n\n")
                f.write(suggested + "\n\n")


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))

    mode = 'test'
    correct_ocr(config[mode])

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
