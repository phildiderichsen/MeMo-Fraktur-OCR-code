"""
tesseract_test.py
Simple test of whether tesseract is working.
"""
import myutils as util
from datetime import datetime
from memoocr.ocr import do_ocr


def main():
    """Run the OCR pipeline."""
    starttime = datetime.now()
    conf = util.get_config()
    cnf = util.Confs(conf)

    do_ocr(cnf.tessconf['imgdir'], cnf.tessconf['outdir'], traineddata_labels=['Fraktur', 'dan', 'frk'])

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
