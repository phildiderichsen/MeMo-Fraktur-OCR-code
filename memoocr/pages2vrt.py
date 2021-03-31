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
    testfolder = os.path.join(conf['intermediatedir'], 'corr_pages', '1870_Lange_AaenOgHavet')

    pages2vrt(testfolder)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def pages2vrt(pagedir):
    """Convert pages to VRT file."""

    def get_pagenum(page: str):
        """Extract page number from page filename."""
        return re.search(r'page_(\d+)', page).group(1)

    pages = [os.path.join(pagedir, p) for p in os.listdir(pagedir)]
    pagenums = [int(get_pagenum(p)) for p in os.listdir(pagedir)]
    vrt_pages = [page2vrt(page, pagenum) for page, pagenum in zip(pages, pagenums)]
    # TODO: When concatenating pages, eliminate hyphenation at end of page.
    print('<text id="{}">\n{}\n</text>'.format(pagedir, "\n".join(vrt_pages)))


def page2vrt(page, pagenum):
    """Transform a single page to VRT format."""
    with open(page, 'r') as infile:
        pagetext = infile.read()
    pagetext = handle_hyphens(pagetext)
    pagelines = [process_line(l, linenum+1, pagenum) for linenum, l in enumerate(pagetext.splitlines())]
    pagetokens = [tokendict for sublist in pagelines for tokendict in sublist]
    vrt_lines = [f'{d["token"]}\t{d["i"]}\t{d["line"]}\t{d["page"]}' for d in pagetokens]
    return '\n'.join(vrt_lines)


def handle_hyphens(text):
    """Eliminate end of line hyphens."""
    return re.sub(r'(\S+)[⸗—-]\n(\S+) ', r'\1\2\n', text)


def process_line(line: str, linenum: int, pagenum: int):
    """Tokenize a line and enumerate the tokens by number on line, line number, and page number."""
    tokens = word_tokenize(line, language='danish')
    return [{'token': tok, 'i': i+1, 'line': linenum, 'page': pagenum} for i, tok in enumerate(tokens)]



if __name__ == '__main__':
    main()
