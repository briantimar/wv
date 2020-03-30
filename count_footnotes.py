from data import DAFIterator

if __name__ == "__main__":
    daf_iter = DAFIterator(logfile="_log_ct.txt")
    outfile = "footnote_counts.txt"
    foot_chars = 0
    main_chars = 0
    foot_words = 0
    main_words = 0
    for i, par in enumerate(daf_iter._paragraphs_raw()):
        if not daf_iter.is_toc(par):
            if daf_iter.is_footnote_link(par):
                pass
            elif daf_iter.is_footnote_text(par):
                foot_chars += len(par.text)
                foot_words += len(par.text.split(' '))
            else:
                main_chars += len(par.text)
                main_words += len(par.text.split(' '))
    
    chars_total = main_chars + foot_chars
    words_total = main_words + foot_words
    with open(outfile, 'w') as f:
        f.write("Main text / footnote size comparisons for DAF\n")
        chars = f"chars: total {chars_total}\n"
        chars += f"\t main text {main_chars} ({100* main_chars / chars_total:.2f} %)\n"
        chars += f"\t footnotes {foot_chars} ({100* foot_chars / chars_total:.2f} %)\n"
        f.write(chars)
        words = f"words: total {words_total}\n"
        words += f"\t main text {main_words} ({100* main_words / words_total:.2f} %)\n"
        words += f"\t footnotes {foot_words} ({100* foot_words / words_total:.2f} %)\n"
        f.write(words)
    

