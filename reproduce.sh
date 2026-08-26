#!/bin/bash
# Rebuilds every figure and number of lw_post.md. Run from this directory.
# Needs: python with torch, numpy, scipy, matplotlib, pillow; lualatex (fontspec, tikz); pdftoppm; pandoc.
set -e
PY=${PY:-$HOME/miniforge3/envs/pivotal/bin/python}

$PY train.py  # the 100 runs behind the strategy table; a no-op while checkpoints/ is populated
for f in posterior_sym asym_compare class_compare gallery search_geometry optimal_geometry opened_cut class_check; do
  echo "== figures/$f.py"
  $PY -m figures.$f
done

cd figures  # the TikZ figures: fonts resolve relative to this directory
for f in setup_diagram naive_cross naive_cases levers opening; do
  lualatex -interaction=batchmode $f.tex > /dev/null
  pdftoppm -png -r 200 -singlefile $f.pdf $f
  $PY -c "from PIL import Image, ImageChops; im = Image.open('$f.png'); bg = Image.new(im.mode, im.size, im.getpixel((0, 0))); im.crop(ImageChops.difference(im, bg).getbbox()).save('$f.png')"
  rm -f $f.aux $f.log
done
cd ..

pandoc lw_post.md -o lw_post.docx --reference-doc=reference.docx  # justified text, centered figures
