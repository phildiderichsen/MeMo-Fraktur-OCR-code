import configparser
import os
import shutil

from datetime import datetime
from memoocr.make_dictionary import make_dic
from memoocr.pdf2img import pdfs2imgs
from memoocr.ocr import do_ocr
from memoocr.correct_ocr import correct_ocr


def main():
    """Run the OCR pipeline."""
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join('config', 'config.ini'))

    conf = config['correct']
    intermediate = os.path.join(conf['intermediatedir'], datetime.now().strftime('%Y-%m-%d'))
    try:
        shutil.copytree(conf['orig_page_dir'], os.path.join(intermediate, 'orig_pages'))
    except FileExistsError:
        pass

    img_dir = os.path.join(intermediate, '1-imgs')

    traineddata_labels = ['fraktur', 'dan', 'frk']

    uncorrected_dirs = [os.path.join(intermediate, f'tess_out_{label}') for label in traineddata_labels]
    uncorrected_dirs.append(os.path.join(intermediate, 'orig_pages'))

    # Set options in the config file for which processing steps to perform.
    if conf.getboolean('run_make_dictionary'):
        make_dic(conf['metadir'])
    if conf.getboolean('run_pdf2img'):
        pdfs2imgs(conf['inputdir'], img_dir, int(conf['split_size']))
    if conf.getboolean('run_ocr'):
        do_ocr(img_dir, intermediate, traineddata_labels)
    if conf.getboolean('correct_ocr'):
        print(uncorrected_dirs)
        correct_ocr(conf, uncorrected_dirs)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
