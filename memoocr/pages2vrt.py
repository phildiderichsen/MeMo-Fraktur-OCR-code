"""
pages2vrt.py
Transform pages in a folder to a VRT file.

- Eliminate hyphenation at EOL.
- Tokenize text.
- Keep word number on line, line and page number as annotation layers.
- When concatenating pages, eliminate hyphenation at end of page.
"""
import configparser
import os
import re

from datetime import datetime
from nltk import word_tokenize


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini'))
    conf = config['DEFAULT']
    testfolder = os.path.join(conf['intermediatedir'], 'corr_pages', '1871_MauT_Deodata')

    vrt = pages2vrt(testfolder)
    print(vrt)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def pages2vrt(pagedir):
    """Convert pages to text represented in VRT format."""

    def get_pagenum(page: str):
        """Extract page number from page filename."""
        return re.search(r'page_(\d+)', page).group(1)

    pages = [os.path.join(pagedir, p) for p in os.listdir(pagedir)]
    pagenums = [int(get_pagenum(p)) for p in os.listdir(pagedir)]
    tokenlists = [page2tokens(page, pagenum) for page, pagenum in zip(pages, pagenums)]
    texttokens = flatten_tokenlists(tokenlists)
    vrt_lines = [f'{d["token"]}\t{d["i"]}\t{d["line"]}\t{d["page"]}' for d in texttokens]
    vrt_text = '<text id="{}">\n{}\n</text>'.format(pagedir, "\n".join(vrt_lines))
    return vrt_text


def page2tokens(page, pagenum):
    """Transform a single page to a list of dicts each containing a token and some annotations."""

    def handle_hyphens(text):
        """Eliminate end of line hyphens."""
        return re.sub(r'(\S+)[⸗—-]\n(\S+) ', r'\1\2\n', text)

    def make_line_tokens(line: str, linenum: int, _pagenum: int):
        """Tokenize a line and enumerate the tokens by number on line, line number, and page number."""
        tokens = word_tokenize(line, language='danish')
        return [{'token': tok, 'i': i + 1, 'line': linenum, 'page': _pagenum} for i, tok in enumerate(tokens)]

    with open(page, 'r') as infile:
        pagetext = infile.read()
    pagetext = handle_hyphens(pagetext)
    pagelines = [make_line_tokens(line, linenum+1, pagenum) for linenum, line in enumerate(pagetext.splitlines())]
    pagetokens = [tokendict for sublist in pagelines for tokendict in sublist]
    return pagetokens


def flatten_tokenlists(tokenlists: list):
    """Flatten lists of page tokens to one big list of text tokens. Eliminate cross-page hyphens."""

    def handle_cross_page_hyphen(page_toks: list, next_page_toks: list):
        """Eliminate hyphenation between page and next_page, if any."""
        if re.match(r'\S+[⸗—-]', page_toks[-1]["token"]) and re.match(r'\S+', next_page_toks[0]["token"]):
            new_prefix = re.sub(r'(\S+)[⸗—-]', r'\1', page_toks[-1]["token"])
            page_toks[-1]["token"] = new_prefix + next_page_toks[0]["token"]
            next_page_toks = next_page_toks[1:]
        return page_toks, next_page_toks

    # For each page: a tuple of the index of the page and the index of the next page (if any)
    indexes = list(range(len(tokenlists)))
    indextuples = list(zip(indexes, indexes[1:]))
    for indextup in indextuples:
        i, j = indextup
        tokenlists[i], tokenlists[j] = handle_cross_page_hyphen(tokenlists[i], tokenlists[j])
    texttokens = [tokendict for sublist in tokenlists for tokendict in sublist]
    return texttokens


if __name__ == '__main__':
    main()
