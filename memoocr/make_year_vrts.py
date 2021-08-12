# Make year_vrts.py
# Concatenate individual texts into VRT files by year.
import os
import re
import shutil


def write_year_vrts(vrt_file, yearcorp_outdir):
    """Concatenate individual texts from vrt_file into VRT files by year.
       We don't assume that the corpora are ordered by year. (But they probably are)."""
    if os.path.isdir(yearcorp_outdir):
        shutil.rmtree(yearcorp_outdir)  # Remove in order to not append the corpora multiple times.
    os.mkdir(yearcorp_outdir)

    with open(vrt_file, 'r') as f:
        vrt_lines = f.read().splitlines()
    current_year = None
    dict_of_year_linelists = dict()
    yearfile_tmpl = 'memo_fraktur_corr_{year}.vrt'

    for line in vrt_lines:
        if '<corpus id' in line:
            continue

        elif '<text id=' in line:
            print(f'Start new corpus and/or text: {line}')
            year = int(re.search(r'<text id="(\d{4})_', line).groups()[0])
            current_year = year
            # touched_years.add(year)
            if year not in dict_of_year_linelists:
                # Start corpus with <corpus> element.
                dict_of_year_linelists[year] = [f'<corpus id="MEMO_FRAKTUR_CORR_{year}">']
            dict_of_year_linelists[year].append(line)

        elif '</corpus>' in line:
            print('End all year corpora')
            for year in dict_of_year_linelists:
                dict_of_year_linelists[year].append(line)
                yearfile = os.path.join(yearcorp_outdir, yearfile_tmpl.format(year=year))
                with open(yearfile, 'a') as f:
                    f.write('\n'.join(dict_of_year_linelists[year]) + '\n')

        else:
            dict_of_year_linelists[current_year].append(line)
