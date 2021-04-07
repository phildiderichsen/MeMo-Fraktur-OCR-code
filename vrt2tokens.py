"""
vrt2tokens.py
Extract each text from a VRT file as a single line of tokens.
(Suitable for processing in Text Tonsorium).
"""

import configparser
import os
import re
from datetime import datetime
from vrt import split_vrt, vrt_text2tokens
from myutils import sorted_listdir


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config', 'config.ini'))
    conf = config['DEFAULT']
    vrt_file = os.path.join(conf['intermediatedir'], 'vrt', 'MEMO_ALL.vrt')
    tt_inputdir = os.path.join(conf['intermediatedir'], 'tt_input')

    try:
        os.makedirs(tt_inputdir)
    except FileExistsError:
        pass

    text_generator = split_vrt(vrt_file)
    for text in text_generator:
        text_id = re.match(r'<text id="([^"]+)">', text.splitlines()[0]).group(1)
        outpath = os.path.join(tt_inputdir, text_id + '.txt')
        vrt_tokens = vrt_text2tokens(text)
        with open(outpath, 'w') as f:
            f.write(' '.join(vrt_tokens))

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
