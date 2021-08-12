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
from myutils import sorted_listdir, tokenize


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

    text_id = os.path.basename(pagedir)
    pages = sorted_listdir(pagedir)
    pagepaths = [os.path.join(pagedir, p) for p in pages]
    pagenums = [int(get_pagenum(p)) for p in pages]
    tokenlists = [page2tokens(page, pagenum, text_id) for page, pagenum in zip(pagepaths, pagenums)]
    texttokens = flatten_tokenlists(tokenlists)
    vrt_lines = [f'{d["token"]}\t{d["i"]}\t{d["line"]}\t{d["page"]}\t{d["text_id"]}' for d in texttokens]
    vrt_text = '<text id="{}">\n{}\n</text>'.format(text_id, "\n".join(vrt_lines))
    return vrt_text


def page2tokens(page, pagenum, text_id):
    """Transform a single page to a list of dicts each containing a token and some annotations."""

    def handle_hyphens(text):
        """Eliminate soft hyphens (\xad) and end of line hyphens."""
        new_text = re.sub(r'\xad *', '', text)
        new_text = re.sub(r'(\S+)[\xad⸗—-]\n(\S+) ', r'\1\2\n', new_text)
        return new_text

    def make_line_tokens(line: str, linenum: int, _pagenum: int, _text_id):
        """Tokenize a line and enumerate the tokens by number on line, line number, and page number."""
        tokens = tokenize(line)
        return [{'token': tok, 'i': i + 1, 'line': linenum, 'page': _pagenum, 'text_id': _text_id} for i, tok in enumerate(tokens)]

    with open(page, 'r') as infile:
        pagetext = infile.read()
    pagetext = handle_hyphens(pagetext)
    pagelines = [make_line_tokens(line, linenum+1, pagenum, text_id) for linenum, line in enumerate(pagetext.splitlines())]
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


def text2vrt(textdir):
    """Convert a whole novel in a text file to VRT format."""
    text_id = os.path.basename(textdir)
    text = [f for f in sorted_listdir(textdir) if f.endswith('.txt')][0]
    textpath = os.path.join(textdir, text)
    tokenlist = text2tokens(textpath, text_id)
    vrt_lines = [f'{d["token"]}\t{d["i"]}\t{d["line"]}\t{d["text_id"]}' for d in tokenlist]
    vrt_text = '<text id="{}">\n{}\n</text>'.format(text_id, "\n".join(vrt_lines))
    return vrt_text


def text2tokens(text, text_id):
    """Transform a single page to a list of dicts each containing a token and some annotations."""

    def make_line_tokens(line: str, linenum: int, _text_id):
        """Tokenize a line and enumerate the tokens by number on line, line number, and page number."""
        tokens = tokenize(line)
        return [{'token': tok, 'i': i + 1, 'line': linenum, 'text_id': _text_id} for i, tok in enumerate(tokens)]

    with open(text, 'r') as infile:
        pagetext = infile.read()
    pagelines = [make_line_tokens(line, linenum+1, text_id) for linenum, line in enumerate(pagetext.splitlines())]
    pagetokens = [tokendict for sublist in pagelines for tokendict in sublist]
    return pagetokens


if __name__ == '__main__':
    main()
