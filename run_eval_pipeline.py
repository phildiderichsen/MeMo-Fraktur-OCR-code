"""
run_eval_pipeline.py
Run evaluation pipeline on gold standard data.
"""
import configparser
import os

from evalocr.make_gold_data import transform_vrt
from datetime import datetime


def main():
    """Run the OCR pipeline."""
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join('config', 'config.ini'))
    conf = config['eval']

    corp_label = conf['fraktur_gold_vrt_label']
    vrt_path = os.path.join(conf['annotated_outdir'], corp_label, corp_label + '.annotated.vrt')

    # Set options in the config file for which processing steps to perform.
    if conf.getboolean('make_gold_vrt'):
        pass
    if conf.getboolean('generate_dataset'):
        transform_vrt(vrt_path)
    if conf.getboolean('analyze_errors'):
        print('Not implemented: analyze_errors')
    if conf.getboolean('write_word'):
        pass

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
