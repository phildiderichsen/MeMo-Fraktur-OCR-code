import configparser
import os

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

    img_dir = os.path.join(conf['intermediatedir'], '1-imgs')
    uncorrected_dir = os.path.join(conf['intermediatedir'], '2-uncorrected')
    corrected_dir = os.path.join(conf['intermediatedir'], '3-corrected')

    traineddata_labels = ['fraktur', 'dan', 'frk']

    # Set options in the config file for which processing steps to perform.
    if conf.getboolean('run_make_dictionary'):
        make_dic(conf)
    if conf.getboolean('run_pdf2img'):
        pdfs2imgs(conf)
    if conf.getboolean('run_ocr'):
        do_ocr(img_dir, conf['intermediatedir'], traineddata_labels)
    if conf.getboolean('correct_ocr'):
        pass  # correct_ocr(conf, uncorrected_dir, corrected_dir)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
