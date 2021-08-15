# README for doing OCR on UCloud

I'm working on running the OCR pipeline on SDU's UCloud infrastructure (cloud.sdu.dk).



## Steps

Log in to cloud.sdu.dk via WAYF.

Add novel image files under "Files" in a directory named "Uploads". Note: This is a major hassle!!!

Upload the sh script provided here to the "Uploads" dir.

Provision a suitable machine from the "Runs" link. The more cores, the better, since the pipeline code is designed for multiprocessing.

chmod the sh script to 755.

Run the sh script in order to set up the machine with all the necessary dependencies.

Make sure the config.ini file has ONLY `run_ocr` set to `yes` under `[correct]`.

Run the pipeline ...



## Notes on performance

The results from UCloud are quite unclear wrt. performance. The processing duration does not scale linearly with the number of pages. Maybe it has to do with page size and/or difficulty. Or maybe differing loads on the UCloud machines. We don't know at this point.

There is more of a pattern when I run on my Mac. Processing seems to go faster with bigger batches.


UCloud:

2 novels (575 pages total) take 42 minutes on a 64 core machine. = 14 pages/min.
1 novel (147 pages total) takes 10 minutes on a 64 core machine. = 15 pages/min.
4 novels (1894 pages total) take 140 minutes on a 64 core machine = 14 pages/min.
4 novels (1917 pages total) take 77 minutes (est. 137) on a 64 core machine = 25 pages/min.
17 novels (5769 pages total) take 1207 minutes (20h7m) (est. 412) on a 64 core machine = 5 pages/min.


My Mac:

1 novel (141 pages total) takes 15 minutes on an 8 core machine = 9 pages/min.
10 novels (4375 pages total) take 336 minutes (est. 486) on an 8 core machine = 13 pages/min.
10 novels (3263 pages total) take 247 minutes (est. 251-363) on an 8 core machine = 13 pages/min.
22 novels (8924 pages total) take 551 minutes (9h11m) on an 8 core machine = 16 pages/min.
)


/Philip Diderichsen, July 2021
