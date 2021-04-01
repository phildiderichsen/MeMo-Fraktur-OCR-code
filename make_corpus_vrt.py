"""
make_corpus_vrt.py
Make VRT files for a whole corpus.
"""
import configparser
import os
from datetime import datetime
from memoocr.pages2vrt import pages2vrt


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config', 'config.ini'))
    conf = config['DEFAULT']

    make_novel_vrts(conf)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


def make_novel_vrts(conf):
    """Make a VRT file for each novel in corpus."""
    novels_dir = os.path.join(conf['intermediatedir'], 'corr_pages')
    vrt_dir = os.path.join(conf['intermediatedir'], 'vrt')

    try:
        os.makedirs(vrt_dir)
    except FileExistsError:
        pass
    novel_ids = sorted(os.listdir(novels_dir))
    novel_dirs = [os.path.join(novels_dir, d) for d in novel_ids]
    for novel_id, novel_dir in zip(novel_ids, novel_dirs):
        outpath = os.path.join(vrt_dir, novel_id + '.vrt')
        with open(outpath, 'w') as f:
            # Process and write novel.
            novel_vrt = pages2vrt(novel_dir)
            f.write(novel_vrt)


if __name__ == '__main__':
    main()
