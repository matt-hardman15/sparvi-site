Inter and Source Serif 4, converted from `site/assets/fonts/*.woff2` to TTF so Pillow can draw the
social preview images (`site/build/social.py`). Not published. Both fonts are licensed under the SIL Open
Font License 1.1 (Inter: The Inter Project Authors; Source Serif 4: Adobe). Reconvert after changing the
site fonts:

    uv run --with brotli python -c "from fontTools.ttLib import TTFont; f = TTFont('site/assets/fonts/inter.woff2'); f.flavor = None; f.save('site/build/fonts/inter.ttf')"
