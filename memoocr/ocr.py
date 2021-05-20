# Core libraries
import configparser
import os
from datetime import datetime
# Image processing
try:
    from PIL import Image
except ImportError:
    import Image
# try to speed up!
import multiprocessing as mp
import pytesseract
#pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
tessdata_dir_config = r'--tessdata-dir "/usr/local/share/tessdata/"'


# OCR process
def process(arg_tuple):
    path, intermediatedir, traineddata_label = arg_tuple
    # for image in [sorted(os.listdir(path))[0]]:  # Use this to only OCR first page in novel/sample.
    for image in os.listdir(path):
        # Create image path with join
        imagepath = os.path.join(path, image)
        print(f'Working on {imagepath}')
        # Convert image to text
        text = pytesseract.image_to_string(imagepath, lang=traineddata_label, config=tessdata_dir_config)
        # Remove obvious noise
        text = text.replace("ſ", "s").replace(",&", ", &")
        """
        text = text.replace("\n\n", "") \
            .replace("\n", " ") \
            .replace("- ", "") \
            .replace(",&", ", &") \
            .replace("ſ", "s")"""

        # get filename by itself with no extension - for later
        name = os.path.split(path)[-1]
        # Create output folder if not exists
        outfolder = os.path.join(intermediatedir, f'tess_out_{traineddata_label}', name)
        # create output folder if not exists
        try:
            os.makedirs(outfolder)
        except FileExistsError:
            pass
        # save to path
        outfile = image.replace('.jpeg', '')
        outpath = os.path.join(outfolder, f'{outfile}_uncorrected.txt')
        with open(outpath, 'w') as f:
            f.write(text)


def do_ocr(img_dir: str, intermediatedir: str, traineddata_labels: list):
    paths = []
    for folder in os.listdir(img_dir):
        paths.append(os.path.join(img_dir, folder))
    for label in traineddata_labels:
        print('Doing Tesseract OCR using traineddata:', label)
        # [('/Users/..../1870_Brosboell_TranensVarsel-s10', '/Users/..../intermediate/2021-03-19', 'fraktur'), ..]
        arg_tuples = list(zip(paths, [intermediatedir] * len(paths), [label] * len(paths)))

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
    traineddata_labels = ['fraktur', 'dan', 'frk']
    do_ocr(img_dir, conf['intermediatedir'], traineddata_labels)

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
