import configparser
from datetime import datetime
import os
# Image processing
try:
    from PIL import Image
except ImportError:
    import Image
from pdf2image import convert_from_path
# TODO: Are these necessary?
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)
# Data flow
import tempfile
import multiprocessing as mp

Image.MAX_IMAGE_PIXELS = None  # otherwise it thinks it's a bomb


def chunked(lst, config, n):
    """
    Yield successive n-sized chunks from lst in order to prevent memory trouble.
    Include config object with each novel in chunk.
    """
    for i in range(0, len(lst), n):
        chunklist = lst[i:i + n]
        yield list(zip(chunklist, [config] * len(chunklist)))


def process(novel_config_tuple):
    """
    Main extraction pipeline

    - It's unclear why but thread_count > 4 causes bottleneck
    - DPI 300 is best quality ratio; higher DPI causes poorer OCR!
    - Format can be set to PNG but it's *much* slower
    """
    novel, conf = novel_config_tuple
    print(f"Working on {novel}")
    # get filepath
    filepath = os.path.join(conf['inputdir'], novel)
    # Create tempfile for images
    with tempfile.TemporaryDirectory() as path:
        print("...converting pdf...")
        images_from_path = convert_from_path(filepath,
                                             thread_count=4,
                                             dpi=300,
                                             output_folder=path,
                                             fmt='jpeg')

        # Save images from path to individual files
        print(f"...saving images for {novel}...")

        # TODO: These should be defined in ONE place ..
        name = novel.replace('.pdf', '')
        outfolder = os.path.join(conf['intermediatedir'], '1-imgs', name)
        # Set page counter
        i = 0
        for image in images_from_path:
            outpath = os.path.join(outfolder, f"page_{i}.jpeg")
            with open(outpath, "w") as out:
                image.save(out)
            image.close()
            i += 1


def pdfs2imgs(conf):
    """
    Main pipe line to convert PDF to JPEG
    """
    # Make list of PDFs in inputdir.
    fs = [f for f in os.listdir(conf['inputdir']) if f.endswith('.pdf')]
    # Make an img folder for each novel. Has to be done here - multiprocessing will choke on it.
    img_dir = os.path.join(conf['intermediatedir'], '1-imgs')
    try:
        os.makedirs(img_dir)
    except FileExistsError:
        print(f"{img_dir} already exists")
    for novel in fs:
        outfolder = os.path.join(img_dir, novel.replace('.pdf', ''))
        try:
            os.mkdir(outfolder)
        except FileExistsError:
            print(f"{outfolder} already exists")

    # Process in chunks equal to the split_size set in options.
    split_size = int(conf['split_size'])
    chunks = list(chunked(fs, conf, split_size))
    print(chunks)
    # Start chunk count at 1
    count = 1
    # Processes equal to split_size; so one CPU per novel
    pool = mp.Pool(processes=split_size)
    # Process in parallel, one chunk at a time to avoid memory leak
    for chunk in chunks:
        print(f"Chunk {count} of {len(chunks)}")
        pool.map(process, chunk)
        count += 1
    pool.close()
    pool.join()


def main():
    starttime = datetime.now()
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))

    mode = 'test'
    pdfs2imgs(config[mode])

    endtime = datetime.now()
    elapsed = endtime - starttime
    print(f"Start: {starttime.strftime('%H:%M:%S')}")
    print(f"End:   {endtime.strftime('%H:%M:%S')}")
    print(f"Elapsed: {elapsed}")


if __name__ == '__main__':
    main()
