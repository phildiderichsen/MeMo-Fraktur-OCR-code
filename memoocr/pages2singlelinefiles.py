import os
import sys
from myutils import readfile


def pages2singlelinefiles(indir, outdir):
    """Transform single page novels to single line novels
    with \v and \f representing line and page breaks"""
    for noveldir in os.listdir(indir):
        full_novel_path = os.path.join(indir, noveldir)
        pages = os.listdir(full_novel_path)
        if len(pages) != 1:
            sys.exit(f'ERROR: No pages or more than 1 page in novel dir {full_novel_path}')
        pagetext = readfile(os.path.join(full_novel_path, pages[0]))
        singleline = pagetext.replace('___PAGEBREAK___ ', '\f').replace('\n', ' \v')
        with open(os.path.join(outdir, f'{noveldir}.txt'), 'w') as f:
            f.write(singleline)
