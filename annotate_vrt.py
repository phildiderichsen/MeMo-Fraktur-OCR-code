"""
annotate_vrt.py
Annotate VRT file containing several novels with e.g. original OCR tokens and difference measures.
"""
import configparser
import os
from datetime import datetime
from vrt import split_vrt
from memoocr.add_vrt_annotations import add_ocr_tokens


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config', 'config.ini'))
    conf = config['DEFAULT']

    vrt_file = os.path.join(conf['intermediatedir'], 'vrt', 'MEMO_ALL.vrt')
    ocr_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    new_vrt = vrt_file.removesuffix('.vrt') + '.annotated.vrt'
    corpus_id = os.path.basename(vrt_file).removesuffix('.vrt')

    text_generator = split_vrt(vrt_file)
    with open(new_vrt, 'w') as outfile:
        outfile.write(f'<corpus id="{corpus_id}">\n')
        for text in text_generator:
            print(text.splitlines()[0])
            outfile.write(add_ocr_tokens(text, ocr_dir))
            outfile.write('\n')
        outfile.write('</corpus>')

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
