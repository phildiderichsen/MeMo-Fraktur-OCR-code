# MeMo-Fraktur-OCR-code

Developing a rule-based/heuristic procedure for correcting OCR data from 19th century fraktur novels.

There are a few different goals in this project:

- Error analysis of baseline OCR output from KB (The Royal Library), as well as OCR output from other sources (i.e. Tesseract).
- Re-OCR'ing the PDFs from KB using Tesseract (using several different pretrained OCR models (traineddata).
- Correcting OCR using regex replacement, context-sensitive rules using alternative OCR from Tesseract, and spelling error detection using SymSpell.
- Error analysis of the corrected OCR vs. the baseline.
- Correcting OCR at scale.



## Project structure

```
MeMo-Fraktur-OCR-code/
├── README.md
├── UCloud
│   ├── README.txt                            # UCloud notes
│   └── ucloud_ocr_provision.sh               # UCloud setup script
├── config
│   ├── config.ini                            # Local settings (git-ignored)
│   ├── config.ini.example.txt                # Example - save as config.ini
│   └── config.ini.ucloud.txt                 # Config for UCloud
├── evalocr                                   # Py package: Evaluation
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   ├── analyze_gold_vrt.cpython-39.pyc
│   │   └── annotate_gold_vrt.cpython-39.pyc
│   ├── analyze_gold_vrt.py
│   └── annotate_gold_vrt.py
├── fulloutput                                # Production output (gitignored)
│   ├── 1-imgs
│   │   ├── 1870_Brosboell_TranensVarsel
│   │   │   ├── page_13.jpeg
│   │   │   ├── page_14.jpeg
│   │   │   ├── ...
│   │   │   ├── page_536.jpeg
│   │   ├── 1870_Dodt_?\206gteOgUægte
│   │   │   ├── page_10.jpeg
│   │   │   ├── ...
│   ├── fraktur_freqs10_correasy_corrhard_symwordcorr
│   │   ├── 1870_Brosboell_TranensVarsel
│   │   │   └── 1870_Brosboell_TranensVarsel.corrected.txt
│   │   ├── 1870_Dodt_?\206gteOgUægte
│   │   │   └── 1870_Dodt_?\206gteOgUægte.corrected.txt
│   │   ├── ...
│   ├── korp
│   │   └── setups
│   │       └── memotest
│   │           ├── corpora
│   │           │   └── encodingscripts
│   │           │       └── encode_MEMO_fraktur_corr.sh
│   │           └── frontend
│   │               └── app
│   │                   └── modes
│   │                       └── memo_frakturcorr_mode.js
│   ├── orig_pages
│   │   ├── 1870_Brosboell_TranensVarsel
│   │   │   └── 1870_Brosboell_TranensVarsel.extr.txt
│   │   ├── ...
│   ├── tess_out_dan
│   │   ├── 1870_Brosboell_TranensVarsel
│   │   │   ├── page_13_uncorrected.txt
│   │   │   ├── ...
│   ├── tess_out_fraktur
│   │   ├── 1873_Johansen_JensSoerensensUngdomshistorie
│   │   │   ├── page_10_uncorrected.txt
│   │   │   ├── ...
│   ├── tess_out_frk
│   │   ├── 1870_Brosboell_TranensVarsel
│   │   │   ├── page_13_uncorrected.txt
│   │   │   ├── ...
│   ├── tt_input
│   │   ├── 1870_Brosboell_TranensVarsel.txt
│   │   ├── ...
│   ├── tt_output
│   │   ├── 1870_Brosboell_TranensVarsel-1.txt-626-step16.conll
│   │   ├── ...
│   ├── vrt
│   │   ├── MEMO_FRAKTUR_CORR.annotated.vrt
│   │   └── MEMO_FRAKTUR_CORR.vrt
│   └── year_vrts
│       ├── memo_fraktur_corr_1870.vrt
│       ├── memo_fraktur_corr_1871.vrt
│       ├── ...
├── intermediate                            # Interim output (git-ignored)
│   ├── 2021-03-19
│   │   ├── 1-imgs
│   │   │   ├── 1870_Brosboell_TranensVarsel-s10
│   │   │   │   └── page_1.jpeg
│   │   │   ├── ...
│   │   ├── 2-uncorrected
│   │   │   ├── 1870_Brosboell_TranensVarsel-s10
│   │   │   │   └── page_1_uncorrected.txt
│   │   │   ├── ...
│   │   ├── 3-corrected
│   │   │   ├── 1870_Brosboell_TranensVarsel-s10
│   │   │   │   └── 1870_Brosboell_TranensVarsel-s10.corrected.txt
│   │   │   ├── ...
│   │   ├── analyses
│   │   │   ├── analyze_gold.orig.txt
│   │   │   ├── analyze_gold.txt
│   │   │   └── eval_ocr_hyphenfix.txt
│   │   ├── corr_pages
│   │   │   ├── 1870_Brosboell_TranensVarsel
│   │   │   │   └── page_18.txt
│   │   │   ├── ...
│   │   ├── eval_datasets
│   │   │   ├── eval_df.csv
│   │   │   └── ocr_eval_df.csv
│   │   ├── orig_pages
│   │   │   ├── 1870_Brosboell_TranensVarsel
│   │   │   │   └── page_18.txt
│   │   │   ├── ...
│   │   ├── orig_pages_corr
│   │   │   ├── 1870_Brosboell_TranensVarsel
│   │   │   │   └── 1870_Brosboell_TranensVarsel.corrected.txt
│   │   │   ├── ...
│   │   ├── ...
│   ├── 2021-05-26
│   │   ├── ...
│   ├── 2021-05-27
│   │   ├── ...
├── memoocr                           # Py package: OCR + corrections
│   ├── __init__.py
│   ├── add_vrt_annotations.py
│   ├── align_ocr.py
│   ├── annotate_corr_vrt.py
│   ├── correct_ocr.py
│   ├── make_corpus_vrt.py
│   ├── make_dictionary.py
│   ├── make_year_vrts.py
│   ├── ocr.py
│   ├── pages2vrt.py
│   └── pdf2img.py
├── myutils
│   ├── __init__.py
│   ├── fraktur_filenames.py
│   ├── novelnames_endpages.py
│   └── novelnames_startpages.py
├── requirements.txt                  # Python requirements
├── run_eval_pipeline.py              # Run evaluation pipeline
├── run_ocr_pipeline.py               # Run correction pipeline
├── tesseract_test.py
├── testdata                          # Data for testing Tesseract
│   └── tesseracttest
│       ├── 1-imgs
│       │   └── 1870_Brosboell_TranensVarsel
│       │       └── page_18.jpeg
│       └── output
│           ├── tess_out_dan
│           │   └── 1870_Brosboell_TranensVarsel
│           │       └── page_18_uncorrected.txt
│           ├── tess_out_fraktur
│           │   └── 1870_Brosboell_TranensVarsel
│           │       └── page_18_uncorrected.txt
│           └── tess_out_frk
│               └── 1870_Brosboell_TranensVarsel
│                   └── page_18_uncorrected.txt
├── venv                             # Py virtual environment
└── vrt2texton_tokens.py             # Extract texts for Text Tonsorium

```

## Installation/dependencies

Code developed in python 3.9 in venv (virtual environment).

- Poppler
    - Installed in `/usr/local/Cellar/poppler/21.02.0` using `brew install poppler`.
- Tesseract 4 
    - Installed in `/usr/local/Cellar/tesseract/4.1.1/bin/tesseract` using `brew install tesseract`
    - Traineddata are placed in `/usr/local/Cellar/tesseract/5.1.0/share/tessdata` (Previously `/usr/local/share/tessdata` ...)
    - Note: On UCloud, the `tessdata` dir is at `/usr/share/tesseract-ocr/4.00/tessdata` ...  
    - frk.traineddata and dan.traineddata must be downloaded from tessdata_fast in order to work: https://github.com/tesseract-ocr/tessdata_fast/raw/master/frk.traineddata, https://github.com/tesseract-ocr/tessdata_fast/raw/master/dan.traineddata
    - fraktur.traineddata can be downloaded from tessdata_best: https://github.com/tesseract-ocr/tessdata_best/raw/master/script/Fraktur.traineddata
- Python modules: See requirements.txt. Install in your virtual environment using `pip install -r requirements.txt`



## Evaluation pipeline

The evaluation pipeline is designed for iteratively developing a heuristic correction procedure with reference to our gold standard.

The pipeline consists of a number of steps. The steps are specified in config.ini.example.txt (which must be adapted to the local system and saved as config.ini). Most of the steps can be omitted as soon as they have been run once.


### Run the evaluation pipeline

To run the evaluation pipeline for the first time:

1. Adapt config.ini.example.txt to your local system, and save it as config.ini.
2. Set all the below steps/parameters (except run_make_dictionary and write_word) to 'yes' in config.ini, and run `run_eval_pipeline.py`.

A bunch of intermediate files will be generated under a dir named after today's date in the `intermediate` dir.

In subsequent runs, many steps, such as generating JPG images from PDFs, can be omitted by setting the relevant parameters to 'no' in config.ini.

On every new date, the full pipeline must be run one first time in order to generate all necessary files. (Alternatively, the dirs can simply be copied over).


### Step: run_make_dictionary

Not used. The frequency dictionaries are currently handmade and stored in the MeMo project's Seafile account.


### Step: run_pdf2img

Generate JPG images from novel page PDFs.


### Step: run_ocr

This step runs Tesseract on the JPG images using a few different traineddata files, cf. the dependencies section.

Output from each tesseract process is saved in a dir named after the traineddata, e.g. 'tess_out_dan', 'tess_out_fraktur', 'tess_out_frk'. Pages from each novel are placed in a dir named after the novel.


### Step: correct_easy

Correct errors that can be corrected in a context-free manner.


### Step: correct_hard

Correct errors with reference to alternative OCR sources.

Most traineddata give overall worse results than the Fraktur.traineddata. However, they perform better on a number of specific error types. For instance, dan.traineddata has high precision in recognizing 'ø' and 'æ'. Transferring 'ø', 'æ' (and other chars) in specific contexts from the 'dan' OCR gives a nice improvement.

More specifically, the corrections are constrained as follows: Correction only happens within suitable correction pairs, i.e. a fraktur token and a token from alternative OCR where the fraktur token contains at least one instance of a specific character (e.g. 't'), and the alternative token contains at least one instance of a specific replacement character (e.g. 'k') in the same position as the fraktur character. For instance, the fraktur token 'tyste', which should have been 'tyske', is corrected using the rule 't' => 'k' - but only the second 't' is replaced. This is because the reference token from the 'frk' OCR is 'tyske', which has a 'k' in the same position as the fraktur token only in the second instance. Correspondingly, the same rule fails to correct the fraktur token 'Stillinger', which should have been 'Skillinger'. This happens because the reference token from the 'frk' OCR is 'Stkillinger' and thus does not have a 'k' in the correct position. Even quite mangled forms from the alternative OCR can be useful. Thus, the 'o' => 'ø' rule applies to the fraktur token 'storste' with reference to the 'dan' OCR to produce the correct form 'største' even when the reference token is 'ftørfte', since 'ø' is in the correct position. Also, if several replacements are relevant, these will all be performed, e.g. a form like 'tysteste' in the context of a reference token like 'kyfkefte' will become 'kyskeste' when subjected to the 't' => 'k' rule.


### Step: sym_wordcorrect

Correct remaining errors using SymSpell at the word level.

This step can be hand-tuned quite a bit by working with the frequency dictionary used by SymSpell. Below are my notes on the different frequency lists I tried out.

- freqs1 = frequency_dict_da_sm.txt (Ross' original unigram list. Common Crawl data?)
- freqs2 = meta/unigrams_brandes_adl.txt (Dorte's unigram freqlist from the Brandes texts and ADL texts in 'Træningskorpus, april 2020')
- freqs3 = unigrams_brandes_adl_ods.txt (freqs2 plus tokens from ODS with freq = 1 if they are not on the freqs2 list)
- freqs4 = unigrams_brandes_adl_da.txt (freqs2, *not lowercased*, plus tokens from freqs1 if they are not on the freqs2 list)
- freqs5 = unigrams_brandes_adl_da_sm.txt (freqs2, *lowercased*, plus tokens from freqs1 if they are not on the freqs2 list)
- freqs6 = unigrams_brandes_adl_da_sm_aa.txt (like freqs5, but with 'å' replaced by 'aa').
- freqs7 = unigrams_brandes_ods_adl_da_sm_aa.txt (like freqs6, but with ODS tokens added (with count = 1) if they are not on freqs6).
- freqs8 = unigrams_brandes_ods6_adl_da_sm_aa.txt (like freqs7, but only ODS tokens at least 6 chars long are added).
- Freqs9 = unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered.txt (ADL words with counts adjusted to match Ross' list; ODS words added if not in ADL; **without** Ross list (by mistake). Inspired by the original SymSpell docs: https://github.com/wolfgarbe/SymSpell#frequency-dictionary: "The frequency_dictionary_en_82_765.txt was created by intersecting the two lists mentioned below (1. Google Books Ngram data and 2. SCOWL - Spell Checker Oriented Word Lists). By reciprocally filtering only those words which appear in both lists are used. Additional filters were applied and the resulting list truncated to ≈ 80,000 most frequent words.".)
- Freqs10 = unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered_names1000.txt (Like freqs9, but with names from the Gold standard texts (that are not on the freqs9 list) added with a frequency of 1000).
- bifreqs1 = bigrams_dict_da_sm.txt (Ross' original bigram list. Common Crawl Data?)
- bifreqs2 = bigrams_brandes_adl.txt (Dorte's bigram freqlist from the Brandes texts and ADL texts in 'Træningskorpus, april 2020')
- bifreqs3 = bigrams_brandes_adlx10.txt (like bifreqs2, but frequencies multiplied by 10 in order to better match the unigram counts)

Observations:

- Freqs10 - with ODS and ADL data, **without** Ross' frequency list (by mistake ..), and with all names from the novels added - performs best.
- Freqs5 performs a tiny bit better than freqs4, which has a slightly lower match percentage, and slightly more error types. Consistent with SymSpell docs which state that SymSpell expects a lowercased frequency dictionary.
- Using bifreqs1, bifreqs2, or bifreqs3 made no difference, but I only did corrections at the word level so far, so this makes sense. Correction of longer stretches of text should be explored (e.g. crudely segmented sentences). 
- Freqs2 has the best performance on the most frequent errors (e.g. only half the ø=o errors compared to freqs1). However, overall performance is far worse than freqs1. (Also, there are many more error types).
- Freqs6 (replacing 'å' with 'aa') gives a slight improvement.
- Freqs8 (limiting ODS tokens to longish words) makes no real difference - if anything, a tiny improvement in the most frequent error types.

Quick and dirty code used to generate combined frequency dict (the code shown generates freqs9):

```python
from collections import defaultdict

adlfreq = '/Users/phb514/my_seafile/Seafile/NorS MeMo Home/MeMo-Fraktur-OCR-cleaning/MeMo-Fraktur-OCR-data/meta/unigrams_brandes_adl.txt'
odsfile = '/Users/phb514/Downloads/ods.freq1.txt'

with open(adlfreq, 'r') as adl:
    adllines = adl.read().splitlines()
with open(odsfile, 'r') as ods:
    odslines = ods.read().splitlines()


odstuples = [line.split() for line in odslines]
odsdict = defaultdict(int)
for token, freq in odstuples:
    if len(token) < 4:
        continue
    token = token.lower().replace('å', 'aa')
    odsdict[token] += int(freq)


adltuples = [line.split() for line in adllines]
adldict = defaultdict(int)
for token, freq in adltuples:
    token = token.lower().replace('å', 'aa')
    adldict[token] += int(freq) * 806

print('len(adldict.items()):', len(adldict.items()))
print("adldict['og']:", adldict['og'])

print('38330072 / 47611:', 38330072 / 47611)

odsplusadl = odsdict.copy()
odsplusadl.update(adldict)


freq_token_tuples = [(v,k) for k,v in odsplusadl.items()]
freq_token_tuples.sort(reverse=True)

freqlist = [f'{k} {v}' for v, k in freq_token_tuples]
print(freqlist[:20])

with open(
        '/Users/phb514/my_seafile/Seafile/NorS MeMo Home/MeMo-Fraktur-OCR-cleaning/MeMo-Fraktur-OCR-data/meta/unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered',
        'w') as out:
    out.write('\n'.join(freqlist))
```



### Step: make_basic_gold_vrt

Make VRT file for CWB from the gold standard (w/o annotations).

Exploring the error/correction data in Korp is very useful. So we need to create a VRT file for Korp from the gold standard. The first step is to create a basic VRT file to which annotations can then be added.

The basic gold VRT is annotated at the token level with:

1. Word number on the line
2. Line number
3. PDF page number
4. Novel ID.

NOTE: In the gold standard input (`Guldstandard.txt`), hypens ("[- ]") are removed.


### Not part of the pipeline: CONLL annotation

In the step annotate_gold_vrt, a number of annotations are added, including word number in sentence, lemma, and PoS. These annotations are computed based on the gold tokens using the NLP pipeline web app "Text Tonsorium" at https://cst.dk/WMS in the following way:

- First, use the script in this project, `vrt2texton_tokens.py`, to transform each novel to a string of tokens without newlines. These files will be saved to `tt_input`, cf. `config.ini`.
- Go to https://cst.dk/WMS.
- In the "Upload input" box, browse to select the files in `tt_input`.
- Click "Specify the required result".
- Select: 
  - Type of content: tokens
  - Format: plain
  - Historical period: late modern
  - Appearance: unnormalised
  - Ambiguity: unambiguous
  - Type of content: segments, tokens, lemmas, PoS-tags
  - Language: Danish
  - Format: CONLL
  - Historical period: late modern.
- Submit, and then select the first workflow suggested.
- Submit, wait ... and finally, click "Download final results".
- The results will be downloaded as a zip file like "502-final.zip". Inside are the CONLL files for each novel, plus some documentation.
- Put the files in tt_output.


### Step: annotate_gold_vrt

In this step, each alternative OCR source as well as the correction output is annotated with the following attributes:

- OCR Token: The OCR output of the given OCR source.
- Levenshtein Distance: Integer representing edit distance.
- Levenshtein Ratio: A word length independent measure of edit distance between 0 and 1.
- CER: Character Error Rate. Implemented as 1 - Lev. ratio.
- Levenshtein Category: Classification of errors into categories such as 'match' (no difference), 'lev_1' (Lev. dist. 1), and 'split_lev_1' (Lev. dist. 1 with spaces involved).
- Substitutions: A representation of errors with <correct>=<error>, e.g. 'o=ø' (correct 'o' became 'ø' in the OCR), '•=t' (a 't' was erroneously introduced in the OCR), 'i=æ+a=e' (several errors).
- In Frequency Dict: 1 if the OCR token is in the frequency dict employed, 0 if not.

Also, annotation layers based on the gold tokens are added: 

Number of word in sentence, lemma, PoS, and whether the gold token is in the frequency list used by SymSpell

Finally, `<sentence>` elements are added to the VRT based on the CONLL output. 


### Step: analyze_errors

Perform error analysis on each OCR source plus on the corrected output.

In this step, the VRT file produced in the last step is transformed into a number of datasets which are then subjected to the same analysis for comparison.

Each OCR source is analyzed (the original KB OCR plus the various Tesseract OCR outputs), and so is the output of the correction process.

Currently, count and percentage breakdowns are produced for the variables Levenshtein Category and Substitutions, and the correctness percentage is broken down by novel using average Levenshtein Ratio and average 'match' percentage. 


### Step: write_korp_configs

Write configuration files for CWB/Korp in order to easily be able to read in the current gold corpus with correction data in Korp for exploration.


### Step: write_word

Produce analysis report in Word. (Not implemented).




## OCR correction pipeline

> Important notes:
> 
> - The novel 1876_Karstens_FrederikFolkekjær was particularly problematic wrt. alignment. Maybe use it as a test case
> - Make sure to exclude novels from ADL (Arkiv for Dansk Litteratur). These exist as clean text and should not be OCR processed.

The correction pipeline is designed for OCR correcting and annotating a whole corpus of full novel PDFs in the Fraktur hand.

The pipeline consists of a number of steps. The steps are specified in config.ini.example.txt (which must be adapted to the local system and saved as config.ini). Most of the steps can be omitted as soon as they have been run once.

Extracted text from each Royal Library (KB) PDF can be found in the directory NorS MeMo Home/KB_extr.txt.

Place each file in its own directory in `fulloutput/orig_pages`, e.g.`fulloutput/orig_pages/1870_Brosboell_TranensVarsel/1870_Brosboell_TranensVarsel.extr.txt`.

> Note: Extracting text from a PDF page by page automatically is a hard task. I found no satisfactory way of doing it in Python. Text from the PDFs from the Royal Library (KB) has to be extracted by hand (copy-paste). 

TODO

- [ ] Implement metadata in the markup.
- [ ] Make trend diagram work. 
- [ ] Implement page numbers in the markup.
- [ ] Improve initial and final page noise handling. (Bounding box detection? Ross?) (See https://www.danvk.org/2015/01/07/finding-blocks-of-text-in-an-image-using-python-opencv-and-numpy.html).
- [ ] Sorting KWIC lines randomly makes more sense in one big corpus ... 

### Run the correction pipeline

The correction pipeline contains similar steps to the evaluation pipeline, but in this case at full scale. The pipeline requires a lot of OCR processing which is best delegated to some kind of high power cloud computing.

### Step: run_make_dictionary

Not used. The frequency dictionaries are currently handmade and stored in the MeMo project's Seafile account.

### Step: run_pdf2img

Split PDFs into separate pages and save each page as jpeg.

The PDF to images step can be run on a dev machine in a couple of hours(?) for about 60 novels.

So, start off by setting only `run_pdf2img` to `yes` in config.ini, and run this step separately using `python3 run_ocr_pipeline.py`.

### Step: run_ocr

Perform OCR with tesseract.

OCR at scale should be done in some kind of batched manner, maybe on different machines. Here are some notes on how I did it - hint: the process was far from perfect.

I did the OCR by running the OCR pipeline on batches of data on SDU's UCloud infrastructure (cloud.sdu.dk), and on my own machine (a Mac).

> Attention: Be careful when managing batches of files and folders ...!

- Log in to cloud.sdu.dk via WAYF.
- Add novel image files under "Files" in a directory named "Uploads". Note: This was a major hassle!!!
- Upload the sh script provided here (in the UCloud directory) to the "Uploads" dir.
- Provision a suitable machine from the "Runs" link on UCloud. The more cores, the better (presumably ..!?), since the pipeline code is designed for multiprocessing.
- chmod the sh script to 755.
- Run the sh script in order to set up the machine with all the necessary dependencies.
- Make sure the config.ini file has ONLY `run_ocr` set to `yes` under `[correct]`.
- Run the pipeline using `python3 run_ocr_pipeline.py`.
- Download the files from UCloud. I used the application minIO. This was a bit of a hassle, too (erratic download behavior with long waits).

Once all files are processed and downloaded, make sure they are all placed correctly in `fulloutput/tess_out_dan` etc.


Notes on performance:

The results from UCloud are quite unclear wrt. performance. The processing duration does not scale linearly with the number of pages. Maybe it has to do with page size and/or difficulty. I don't know at this point.

There is more of a pattern when I run on my Mac. Processing seems to go faster with bigger batches.


64 core UCloud VMs:

- 2 novels (575 pages total) take 42 minutes = 14 pages/min.
- 1 novel (147 pages total) takes 10 minutes = 15 pages/min.
- 4 novels (1894 pages total) take 140 minutes = 14 pages/min.
- 4 novels (1917 pages total) take 77 minutes (est. 137) = 25 pages/min.
- 17 novels (5769 pages total) take 1207 minutes (20h7m) (est. 412) = 5 pages/min.


My 8 core Mac:

- 1 novel (141 pages total) takes 15 minutes = 9 pages/min.
- 10 novels (4375 pages total) take 336 minutes (est. 486) = 13 pages/min.
- 10 novels (3263 pages total) take 247 minutes (est. 251-363) = 13 pages/min.
- 22 novels (8924 pages total) take 551 minutes (9h11m) = 16 pages/min.
)



### Cleaning steps: correct_easy, correct_hard, sym_wordcorrect

Clean OCR from tesseract using the same steps as in the evaluation pipeline.

The cleaning steps require all the different Tesseract outputs to be present in `fulloutput/tess_out_dan` etc.

To run the pipeline with the cleaning steps only, set `correct_easy`, `correct_hard`, and `sym_wordcorrect` to `yes` in config.ini, and run `python3 run_ocr_pipeline.py`.



### Step: make_basic_gold_vrt

Similar to above. But I had to make specialized processing for whole texts rather than pages.



### Not part of the pipeline: CONLL annotation

See above.



### Step: annotate_corr_vrt

In this step, CONLL annotation layers based on the corrected tokens are added: 

Number of word in sentence, lemma, PoS, and whether the corrected token is in the frequency list used by SymSpell.

Finally, `<sentence>` elements are added to the VRT based on the CONLL output. 

Also, the following attributes from the KB OCR are added:

- OCR Token: The OCR output of the KB PDF.
- Levenshtein Distance: Integer representing edit distance between Fraktur and KB OCR.
- Levenshtein Ratio: A word length independent measure of edit distance between 0 and 1.
- CER: Character Error Rate. Implemented as 1 - Lev. ratio.
- Levenshtein Category: Classification of errors into categories such as 'match' (no difference), 'lev_1' (Lev. dist. 1), and 'split_lev_1' (Lev. dist. 1 with spaces involved).
- Substitutions: A representation of errors with <correct>=<error>, e.g. 'o=ø' (correct 'o' became 'ø' in the OCR), '•=t' (a 't' was erroneously introduced in the OCR), 'i=æ+a=e' (several errors).
- In Frequency Dict: 1 if the OCR token is in the frequency dict employed, 0 if not.

In the correction pipeline, the tesseract OCR sources are not added for now.


### Step: add_metadata

Add metadata from the file MeMo-Fraktur-OCR-code/metadata.tsv (copied from romankorpus_metadata_onedrive.xlsx).

### Step: make_yearcorpora

Run a function that creates individual year corpora from the big unified corpus file, and generates the necessary configuration files for Korp.



### Step: export_corpora

Copy the corpora and configuration files to a local Korp x Docker setup directory. 


### Build local Korp in Docker

After running the pipeline, including the `export_corpora` step to place corpora and config files in the correct locations, the corpora can be deployed to a local instance of Korp in Docker.

Build the Korp backend and frontend, and index the corpora in CWB (Corpus Workbench) in the backend. Use the `memotest_feature` branch of the infrastructure repo for this. (https://github.com/kuhumcst/infrastructure/tree/memotest_feature/korp/setups/memotest).


```bash
cd setups/memotest
git checkout memotest_feature

# Edit settings.modeConfig in setups/memotest/frontend/app/config.js. localekey: "memo_fraktur_corrected", mode: "memo_frakturcorr".
# Edit the translation files in setups/clarin/frontend/app/translations so that all new labels have a translation.

# Rebuild the Docker container, and index the corpora in CWB (Corpus Workbench)
cd setups/memotest
docker-compose down ; docker-compose up -d --build ; docker-compose exec backend bash /opt/corpora/encodingscripts/encode_MEMO_fraktur_corr.sh
```

The frontend and backend are now available locally at http://localhost:9111 and http://localhost:1234, respectively.


### Deploy the year corpora in CLARIN's Korp

In production, the MeMo corpora will be integrated in the CLARIN.dk Korp setup. Follow these steps to upload the encoded corpora and the necessary configuration files.

1. Generate and index the corpora as described above.
2. Copy the encoded corpora to the CLARIN backend: `scp -r /Users/phb514/my_git/infrastructure/korp/setups/memotest/corpora/data/memo_fraktur_corr_* phb514@nlpkorp01fl:/opt/corpora/data/ ; scp -r /Users/phb514/my_git/infrastructure/korp/setups/memotest/corpora/registry/memo_fraktur_corr_* phb514@nlpkorp01fl:/opt/corpora/registry/`
3. Check that the corpora exist in the backend: https://alf.hum.ku.dk/korp/backend.
4. Switch to the shadowmaster git branch and merge master.
5. Copy the mode file for the corpora: `scp /Users/phb514/my_git/infrastructure/korp/setups/memotest/frontend/app/modes/memo_frakturcorr_mode.js phb514@nlpkorp01fl:/opt/corpora/infrastructure/korp/setups/clarin/frontend/app/modes/memo_frakturcorr_mode.js`
6. Edit `settings.modeConfig` in `setups/clarin/frontend/app/config.js`. `localekey: "memo_fraktur_corrected", mode: "memo_frakturcorr"`.
7. Edit the translation files so that all new labels have a translation.
8. Rebuild the CLARIN Docker container: `cd setups/clarin ; sudo docker-compose down ; sudo docker-compose up -d --build`. 

# Notes by date

## 6.5.2022

- Exchange the kb OCR with the Fraktur OCR in the selective correction step - with a bunch of new replacement categories. This brings accuracy to an unprecedented 97.81.
- The frk OCR source can then also be omitted from the selective correction step, still yielding 97.81.
- Hack precision/recall code to be able to show prec/rec for the new kb data.
- Swapping the Texton Tesseract v. 5 ("med ø") OCR in instead of the Tesseract "dan" OCR makes exactly zero difference - still 97.81. But slightly different distribution of remaining errors (e.g. ø=o = 12 vs. 10 with the "dan" data).
- If a few more replacements are added to "dan", "Fraktur" can be omitted while still yielding 97.60.
- Tweaking Symspell so that a selection of false positive corrections are cancelled brings WER back up to 97.80 (but of course not necessarily a very general effect).


## 4.5.2022

- Running with new Fraktur scans from KB as base OCR, and all three correction steps, yields the best results yet (97.62). Note: The "KB" dataset was still used in the selective correction step - even though the "KB" dataset was used as base OCR. (=> Maybe use "Fraktur" for the third OCR source in selective correction instead?)
- Just Easy corrections + SymSpell yield better results than all three steps with Fraktur as base OCR (97.22).
- tess_out_frk was pointed to a copy of /Users/phb514/Downloads/tess_out_texton (output from Texton Tesseract v. 5 "med ø") instead of the "frk" model. This made no difference for results. (Made a flat folder with renamed images to OCR with Texton (1-imgs-renamed) for this purpose). 
- Untouched new Fraktur scans from KB are much better (95.83) than untouched Tesseract Fraktur scans (89.54).

