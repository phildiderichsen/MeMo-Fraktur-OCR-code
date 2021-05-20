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
    intermediate = conf['intermediatedir']

    img_dir = os.path.join(intermediate, '1-imgs')
    uncorrected_dir = os.path.join(intermediate, '2-uncorrected')
    corrected_dir = os.path.join(intermediate, '3-corrected')

    traineddata_labels = ['fraktur', 'dan', 'frk']

    uncorrected_dirs = [os.path.join(intermediate, f'tess_out_{label}') for label in traineddata_labels]
    uncorrected_dirs.append(os.path.join(intermediate, 'orig_pages'))

    # Set options in the config file for which processing steps to perform.
    if conf.getboolean('run_make_dictionary'):
        make_dic(conf)
    if conf.getboolean('run_pdf2img'):
        pdfs2imgs(conf)
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
