# Core libraries
import configparser
import os
from datetime import datetime
# Image processing
try:
    from PIL import Image
except ImportError:
    import Image
import myutils as util
# try to speed up!
import multiprocessing as mp
import pytesseract
from itertools import product

conf = util.get_config()
pth = util.Confs(conf)

pytesseract.pytesseract.tesseract_cmd = pth.tessconf['tess_bin']
tessdata_dir_config = fr'''--tessdata-dir "{pth.tessconf['tessdata_dir']}"'''


def process(arg_tuple):
    """OCR process."""
    path, outdir, traineddata_label = arg_tuple
    # for image in [sorted(util.sorted_listdir(path))[0]]:  # Use this to only OCR first page in novel/sample.
    for image in util.sorted_listdir(path):
        # Create image path with join
        imagepath = os.path.join(path, image)
        print(f'Working on {imagepath}')
        # Convert image to text
        text = pytesseract.image_to_string(imagepath, lang=traineddata_label, config=tessdata_dir_config)
        # Remove obvious noise
        text = text.replace("ſ", "s").replace(",&", ", &")
        # get filename by itself with no extension - for later
        name = os.path.split(path)[-1]
        # Create output folder if not exists
        outfolder = os.path.join(outdir, f'tess_out_{traineddata_label}', name)
        # create output folder if not exists
        util.safe_makedirs(outfolder)
        # save to path
        outfile = image.replace('.jpeg', '')
        outpath = os.path.join(outfolder, f'{outfile}_uncorrected.txt')
        with open(outpath, 'w') as f:
            f.write(text)


def do_ocr(img_dirs: list, outdir: str, traineddata_labels: list):
    """Do OCR using multiprocessing."""
    arg_tuples = list(product(img_dirs, [outdir], traineddata_labels))
    n_processes = mp.cpu_count() - 2 if mp.cpu_count() > 2 else mp.cpu_count()
    pool = mp.Pool(processes=n_processes)
    pool.map(process, arg_tuples)
    pool.close()


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini'))

    conf = config['DEFAULT']
    img_dir = os.path.join(conf['intermediatedir'], '1-imgs')
    traineddata_labels = ['Fraktur', 'dan', 'frk']
    do_ocr(img_dir, conf['intermediatedir'], traineddata_labels)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
