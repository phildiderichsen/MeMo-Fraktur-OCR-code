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
# TODO: These are apparently not necessary ..?
#  If they are, they should be made into options in config.ini.
#pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"
#tessdata_dir_config = r'--tessdata-dir "/usr/local/share/tessdata/"'


# OCR process
def process(path_config_tuple):
    path, conf = path_config_tuple
    # for image in [sorted(os.listdir(path))[0]]:  # Use this to only OCR first page in novel/sample.
    for image in os.listdir(path):
        print(f'Working on {image}')
        # Create image path with join
        imagepath = os.path.join(path, image)
        # Convert image to text
        text = pytesseract.image_to_string(imagepath, lang="fraktur")
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
        outfolder = os.path.join(conf['intermediatedir'], '2-uncorrected', name)
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


def do_ocr(conf):
    paths = []
    img_dir = os.path.join(conf['intermediatedir'], '1-imgs')
    for folder in os.listdir(img_dir):
        paths.append(os.path.join(img_dir, folder))
    path_config_tuples = list(zip(paths, [conf] * len(paths)))

    n_processes = mp.cpu_count() - 2 if mp.cpu_count() > 2 else mp.cpu_count()
    pool = mp.Pool(processes=n_processes)
    pool.map(process, path_config_tuples)
    pool.close()


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config', 'config.ini'))

    mode = 'test'
    do_ocr(config[mode])

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()