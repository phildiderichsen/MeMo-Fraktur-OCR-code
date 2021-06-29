# MeMo-Fraktur-OCR-code

Developing a rule-based/heuristic procedure for correcting OCR data from 19th century fraktur novels.

There are a few different goals in this project:

- Error analysis of baseline OCR output from KB (The Royal Library), as well as OCR output from other sources (= Tesseract).
- Re-OCR'ing the PDFs from KB using Tesseract (possibly using several different models).
- Correcting OCR using regex replacement, spelling error detection, and possibly custom character level n-gram embedding similarity.
- Error analysis of the corrected OCR vs. the baseline.



## Project structure

```
MeMo-Fraktur-OCR-code
├── README.md
├── config
│   ├── config.ini                                # Local settings (git-ignored)
│   ├── config.ini.example.txt                    # Example - save as config.ini
│   ├── encode_memo_frakturgold.txt
│   └── memo_frakturgold_mode.txt
├── evalocr                                       # Py package: Evaluation
│   ├── __init__.py
│   ├── analyze_gold_vrt.py
│   └── annotate_gold_vrt.py
├── intermediate                                  # Interim output (git-ignored)
│   └── 2021-06-01
│       ├── 1-imgs
│       │   ├── 1870_Brosboell_TranensVarsel-s10
│       │   │   └── page_1.jpeg
│       ├── analyses
│       │   ├── fraktur_freqs9_correasy.txt
│       │   └── fraktur_freqs9_correasy_corrhard_symwordcorr.txt
│       ├── fraktur_freqs9_correasy
│       │   ├── 1870_Brosboell_TranensVarsel-s10
│       │   │   └── 1870_Brosboell_TranensVarsel-s10.corrected.txt
│       ├── fraktur_freqs9_correasy_corrhard_symwordcorr
│       │   ├── 1870_Brosboell_TranensVarsel-s10
│       │   │   └── 1870_Brosboell_TranensVarsel-s10.corrected.txt
│       ├── gold_pages
│       │   ├── 1870_Brosboell_TranensVarsel
│       │   │   └── page_18.txt
│       ├── orig_pages
│       │   ├── 1870_Brosboell_TranensVarsel
│       │   │   └── page_18.txt
│       ├── tess_out_dan
│       │   ├── 1870_Brosboell_TranensVarsel-s10
│       │   │   └── page_1_uncorrected.txt
│       ├── tess_out_fraktur
│       │   ├── 1870_Brosboell_TranensVarsel-s10
│       │   │   └── page_1_uncorrected.txt
│       ├── tess_out_frk
│       │   ├── 1870_Brosboell_TranensVarsel-s10
│       │   │   └── page_1_uncorrected.txt
│       └── vrt
│           ├── fraktur_freqs9_correasy
│           │   ├── MEMO_FRAKTUR_GOLD.annotated.vrt
│           │   ├── MEMO_FRAKTUR_GOLD.vrt
│           │   ├── encode_MEMO_fraktur_gold.sh
│           │   └── memo_frakturgold_mode.js
│           └── fraktur_freqs9_correasy_corrhard_symwordcorr
│               ├── MEMO_FRAKTUR_GOLD.annotated.vrt
│               ├── MEMO_FRAKTUR_GOLD.vrt
│               ├── encode_MEMO_fraktur_gold.sh
│               └── memo_frakturgold_mode.js
├── memoocr                                   # Py package: OCR + corrections
│   ├── __init__.py
│   ├── add_vrt_annotations.py
│   ├── align_ocr.py
│   ├── correct_ocr.py
│   ├── make_corpus_vrt.py
│   ├── make_dictionary.py
│   ├── ocr.py
│   ├── pages2vrt.py
│   └── pdf2img.py
├── myutils
├── requirements.txt                          # Python requirements
├── run_eval_pipeline.py                      # Run evaluation pipeline
├── run_ocr_pipeline.py                       # Run correction pipeline
└── vrt2texton_tokens.py                      # Extract texts for Text Tonsorium

```

## Installation/dependencies

Code developed in python 3.9 in venv (virtual environment).

- Poppler
    - Installed in `/usr/local/Cellar/poppler/21.02.0` using `brew install poppler`.
- Tesseract 4 
    - Installed in `/usr/local/Cellar/tesseract/4.1.1/bin/tesseract` using `brew install tesseract`
    - Traineddata are placed in `/usr/local/share/tessdata`
    - frk.traineddata and dan.traineddata must be downloaded from tessdata_fast in order to work: https://github.com/tesseract-ocr/tessdata_fast/blob/master/frk.traineddata, https://github.com/tesseract-ocr/tessdata_fast/blob/master/dan.traineddata
    - fraktur.traineddata can be downloaded from tessdata_best: https://github.com/tesseract-ocr/tessdata_best/blob/master/script/Fraktur.traineddata
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
- Freqs9 = unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered.txt (ADL words with counts adjusted to match Ross' list; ODS words added if not in ADL; Ross list added, but only words in ODS. Inspired by the original SymSpell docs: https://github.com/wolfgarbe/SymSpell#frequency-dictionary: "The frequency_dictionary_en_82_765.txt was created by intersecting the two lists mentioned below (1. Google Books Ngram data and 2. SCOWL - Spell Checker Oriented Word Lists). By reciprocally filtering only those words which appear in both lists are used. Additional filters were applied and the resulting list truncated to ≈ 80,000 most frequent words.".)
- Freqs10 = unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered_names1000.txt (Like freqs9, but with names from the Gold standard texts (that are not on the freqs9 list) added with a frequency of 1000).
- bifreqs1 = bigrams_dict_da_sm.txt (Ross' original bigram list. Common Crawl Data?)
- bifreqs2 = bigrams_brandes_adl.txt (Dorte's bigram freqlist from the Brandes texts and ADL texts in 'Træningskorpus, april 2020')
- bifreqs3 = bigrams_brandes_adlx10.txt (like bifreqs2, but frequencies multiplied by 10 in order to better match the unigram counts)

Observations:

- Freqs10 - with ODS and ADL data, and with Ross' frequency list filtered on ODS, and with all names from the novels added - performs best.
- Freqs5 performs a tiny bit better than freqs4, which has a slightly lower match percentage, and slightly more error types. Consistent with SymSpell docs which state that SymSpell expects a lowercased frequency dictionary.
- Using bifreqs1, bifreqs2, or bifreqs3 made no difference, but I only did corrections at the word level so far, so this makes sense. Correction of longer stretches of text should be explored (e.g. crudely segmented sentences). 
- Freqs2 has the best performance on the most frequent errors (e.g. only half the ø=o errors compared to freqs1). However, overall performance is far worse than freqs1. (Also, there are many more error types).
- Freqs6 (replacing 'å' with 'aa') gives a slight improvement.
- Freqs8 (limiting ODS tokens to longish words) makes no real difference - if anything, a tiny improvement in the most frequent error types.

Quick and dirty code used to generate combined frequency dict (the code shown generates freqs9):

```python
from collections import defaultdict

rossfreq = '/Users/phb514/my_seafile/Seafile/NorS MeMo Home/MeMo-Fraktur-OCR-cleaning/MeMo-Fraktur-OCR-data/meta/frequency_dict_da_sm.txt'
adlfreq = '/Users/phb514/my_seafile/Seafile/NorS MeMo Home/MeMo-Fraktur-OCR-cleaning/MeMo-Fraktur-OCR-data/meta/unigrams_brandes_adl.txt'
odsfile = '/Users/phb514/Downloads/ods.freq1.txt'

with open(rossfreq, 'r') as rf:
    rflines = rf.read().splitlines()
with open(adlfreq, 'r') as adl:
    adllines = adl.read().splitlines()
with open(odsfile, 'r') as ods:
    odslines = ods.read().splitlines()

"""
Hvad skal der ske?
Ross' frekvensliste skal gælde hvis ordet ikke findes i ADL.
Hvis ordet findes i ADL, skal det indsættes i stedet, med frekvensen korrigeret op i samme størrelsesorden som Ross' liste.
"""


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
    # adldict[token] += int(freq) * 806
    token = token.lower().replace('å', 'aa')
    adldict[token] += int(freq) * 806

print('len(adldict.items()):', len(adldict.items()))
print("adldict['og']:", adldict['og'])

print('38330072 / 47611:', 38330072 / 47611)

odsplusadl = odsdict.copy()
odsplusadl.update(adldict)


rosstuples = [line.split() for line in rflines]
rossdict = defaultdict(int)
for token, freq in rosstuples:
    token = token.lower().replace('å', 'aa')
    rossdict[token] += int(freq)

print('len(rossdict.items()):', len(rossdict.items()))
print("rossdict['og']:", rossdict['og'])



combineddict = odsplusadl.copy()
combineddict.update(odsplusadl)
print('len(combineddict.items()):', len(combineddict.items()))


# Remove words that are not in ODS + ADL
odsadl_filtered_combineddict = [(k, v) for k, v in combineddict.items() if k in odsplusadl]

freq_token_tuples = [(v,k) for k,v in odsadl_filtered_combineddict]
freq_token_tuples.sort(reverse=True)

freqlist = [f'{k} {v}' for v, k in freq_token_tuples]
print(freqlist[:20])

with open('/Users/phb514/my_seafile/Seafile/NorS MeMo Home/MeMo-Fraktur-OCR-cleaning/MeMo-Fraktur-OCR-data/meta/unigrams_brandes_ods_adl_da_sm_aa_odsadlfiltered.txt', 'w') as out:
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




## OCR correction

Work in progress ...

Take PDFs of Fraktur novels through an OCR pipeline from PDF to cleaned OCR text.

- Split PDFs into separate pages and save each page as jpeg.
- Perform OCR with tesseract.
- Clean OCR from tesseract using regex/SymSpell/character n-gram embeddings?
  - The cleaned output is to be tokenized and used as the token layer for all subsequent annotations. The original OCR text is to be aligned to this token layer for later reference.

The different steps can be bypassed as per the choices in config.ini.

Run:

Create a config.ini with relevant options and local paths (see `config.ini.example.txt`). Then:

```
python3 run_ocr_pipeline.py
```

