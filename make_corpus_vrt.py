"""
make_corpus_vrt.py
Make VRT file(s) for a whole corpus.
"""
import configparser
import os
from datetime import datetime
from memoocr.pages2vrt import pages2vrt
from myutils import sorted_listdir


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config', 'config.ini'))
    conf = config['DEFAULT']

    make_novel_vrt(conf)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def make_novel_vrt(conf, corpus_id='MEMO_ALL'):
    """Make a single VRT for all novels in corpus."""
    novels_dir = os.path.join(conf['intermediatedir'], 'corr_pages')
    vrt_dir = os.path.join(conf['intermediatedir'], 'vrt')

    try:
        os.makedirs(vrt_dir)
    except FileExistsError:
        pass
    novel_ids = sorted_listdir(novels_dir)
    novel_dirs = [os.path.join(novels_dir, d) for d in novel_ids]
    outpath = os.path.join(vrt_dir, corpus_id + '.vrt')
    with open(outpath, 'w') as f:
        f.write(f'<corpus id="{corpus_id}">\n')
        for novel_id, novel_dir in zip(novel_ids, novel_dirs):
            # Process and write novel.
            novel_vrt = pages2vrt(novel_dir)
            f.write(novel_vrt)
            f.write('\n')
        f.write('</corpus>')


if __name__ == '__main__':
    main()
