"""
make_corpus_vrt.py
Make VRT file(s) for a whole corpus.
"""
import configparser
import os
from datetime import datetime
from memoocr.pages2vrt import pages2vrt, text2vrt
from myutils import sorted_listdir

from memoocr import ROOT_PATH


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_PATH, 'config', 'config.ini'))
    conf = config['DEFAULT']

    novels_dir = os.path.join(conf['intermediatedir'], 'corr_pages')
    vrt_dir = os.path.join(conf['intermediatedir'], 'vrt')
    corpus_id = 'MEMO_FRAKTUR_GOLD'

    novels_vrt_gen = generate_novels_vrt(novels_dir, corpus_id)
    write_novels_vrt(novels_vrt_gen, os.path.join(vrt_dir, corpus_id + '.vrt'))

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def generate_novels_vrt(novels_dir, corpus_id, mode='pages'):
    """Generator that yields the lines of a VRT file with all novels in a corpus."""
    novel_ids = sorted_listdir(novels_dir)
    novel_dirs = [os.path.join(novels_dir, d) for d in novel_ids]
    yield f'<corpus id="{corpus_id}">' + '\n'
    for novel_id, novel_dir in zip(novel_ids, novel_dirs):
        # Process and write novel.
        if mode == 'text':
            novel_vrt = text2vrt(novel_dir)
        else:
            novel_vrt = pages2vrt(novel_dir)
        yield novel_vrt + '\n'
    yield '</corpus>' + '\n'


def write_novels_vrt(vrt_generator, outpath):
    """Write a single VRT for all novels in corpus."""
    with open(outpath, 'w') as f:
        for line in vrt_generator:
            f.write(line)


if __name__ == '__main__':
    main()
