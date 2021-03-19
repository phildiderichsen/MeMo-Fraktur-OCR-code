import configparser

from datetime import datetime
from memoocr.make_dictionary import make_dic
from memoocr.pdf2img import pdfs2imgs
from memoocr.ocr import do_ocr
from memoocr.correct_ocr import correct_ocr


def main():
    """Run the OCR pipeline."""
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read('config.ini')

    conf = config['DEFAULT']
    # Set options in the config file for which processing steps to perform.
    if conf.getboolean('run_make_dictionary'):
        make_dic(conf)
    if conf.getboolean('run_pdf2img'):
        pdfs2imgs(conf)
    if conf.getboolean('run_ocr'):
        do_ocr(conf)
    if conf.getboolean('correct_ocr'):
        correct_ocr(conf)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
