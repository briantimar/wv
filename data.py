import requests
from bs4 import BeautifulSoup
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import word_tokenize

## the gibbon utf-8 txt
# target = "https://www.gutenberg.org/files/25717/25717-0.txt"
# fname = "data/gibbon_dat.txt" 

## html gibbon
target = "https://www.gutenberg.org/files/25717/25717-h/25717-h.htm"
fname = "data/gibbon_dat.html"

## full text of the lord of the rings

# target = "http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_1__en.htm"
# fname = "data/lotr.txt"

class NoFootnoteError(ValueError):
    pass

def download_target_url(target, fname):
    """Download text from the given url to fname."""
    r = requests.get(target)
    if r.ok:
        with open(fname, 'w') as f:
            f.write(r.text)
        print(f"Wrote {len(r.text)} chars to {fname}.")

class DAFIterator:
    """Interface to the gibbon html file in local dir."""

    #paragraph element at which the main text starts
    PAR_START = 481

    def __init__(self, fname="data/gibbon_daf.html", logfile="log.txt"):
        self.fname = fname
        self.logfile = logfile
        self.parsed = None
        self.load_parsed()

        self._reset()

    def _reset(self):
        self._current_body_par = None
        self.main_count = 0
        self.footnote_count = 0
        self._logbuf = []

    def log(self, s):
        self._logbuf.append(s)



    def _make_log_header(self):
        header = [f"linking text in {self.fname}"]
        totalct = self.main_count + self.footnote_count
        header.append(f"total chars: {totalct}")
        header.append(f"maintext chars: {self.main_count} ({100 * self.main_count / totalct:.4f}%)")
        header.append(f"footnote chars: {self.footnote_count} ({100 * self.footnote_count / totalct:.4f}%)")
        header.append("------")
        self._logbuf = header + self._logbuf

    def flush_log(self):
        self._make_log_header()
        with open(self.logfile, 'w') as f:
            for s in self._logbuf:
                f.write(s + "\n")

        self._logbuf = []

    def load_parsed(self):
        """Load parsed beautifulsoup object holding the full html"""
        with open(self.fname) as f:
            self.parsed = BeautifulSoup(f.read(), features="html.parser")

    def _paragraphs_raw(self):
        """Iterator over main-text paragraph elements; this includes footnotes."""
        for par in self.parsed.find_all("p")[self.PAR_START:]:
            yield par

    def is_footnote_text(self, par):
        """Checks whether an element contains footnote text."""
        return (par is not None) and ("foot" in par.attrs.get("class", []))
    
    def is_footnote_link(self, par):
        """Checks whether an element is a link adjacent to footnote text."""
        return self.is_footnote_text(par.find_next_sibling('p'))
    
    def is_footnote(self, par):
        """Checks whether a paragraph element is part of a footnote."""
        if par.find_next_sibling('p') is None:
            return False
        return self.is_footnote_text(par) or self.is_footnote_link(par)

    def is_toc(self, par):
        """Checks whether a paragraph is part of a table of contents."""
        return "toc" in par.attrs.get("class", [])

    def _main_paragraphs_raw(self):
        """Main-text paragraphs only."""
        for par in self._paragraphs_raw():
            #wasteful...
            if (not self.is_toc(par)) and (not self.is_footnote(par)):
                self._current_body_par = par
                yield par

    def _get_footnote_par(self, id):
        """Returns paragraph element corresponding to the given id."""
        start = self._current_body_par
        if start is None:
            start = self.parsed
        link = start.find_next(id=id)
        if link is None:
            raise NoFootnoteError(f"Could not find id {id}")
        foot_par = link.parent.find_next_sibling('p')
        if not self.is_footnote_text(foot_par):
            raise NoFootnoteError(f"Failed to find adjacent link paragraph for footnote {id}.")
        return foot_par

    def linked_text_paragraphs(self):
        """Walk over pararaphs in the main text. If a footnote link is found, jump to that paragraph, 
        then back to the main text.
        Returns: iterator over paragraph-sized strings"""
        for par in self._main_paragraphs_raw():
            par_links = par.find_all('a')
            if len(par_links) == 0:
                self.main_count += len(par.text)
                yield par.text
            else:
                for el in par.contents:
                    if el.name is None:
                        #this is plain text
                        self.main_count += len(str(el))
                        yield str(el)
                    elif el.name == "a" and "href" in el.attrs:
                        id = el["href"].lstrip('#')
                        try:
                            foot_par = self._get_footnote_par(id)
                        except NoFootnoteError:
                            self.log(f"Could not find footnote for {id}, skipping.")
                        self.footnote_count += len(foot_par.text)
                        yield foot_par.text

class DAFWords:

    def __init__(self, textfile="data/gibbon_daf_linked.txt"):
        self.textfile = textfile

    punctuation=",.:?;-*\&\'\""

    def tokenize_generic(self, tokenizer, N=None, drop_punctuation=True):
        """Tokenize on line-by-line basis."""
        ct, done = 0, False
        with open(self.textfile) as f:
            for ln in f.readlines():
                if done:
                    break
                ln = ln.replace("(return)", "")
                for token in tokenizer(ln.strip()):
                    if not done:
                        if (not drop_punctuation) or (token not in self.punctuation):
                            yield token
                            ct += 1
                    if (N is not None ) and ct == N:
                        done = True
    
    def tokenize(self, N=None, drop_punctuation=True, lower=True):
        """tokenize using the nltk default (ptb + a 'punkt' sentence tokenizer)"""
        for tok in self.tokenize_generic(word_tokenize, N=N, drop_punctuation=drop_punctuation):
            if lower:
                tok = tok.lower()
            yield tok

def write_linked_text():

    outfile = "data/gibbon_daf_linked.txt"
    logfile = "data/gibbon_daf_linked_log.txt"
    wordIter = DAFIterator(logfile=logfile)

    with open(outfile, 'w') as f:
        for i,s in enumerate(wordIter.linked_text_paragraphs()):
            if i%1000 == 0:
                print(i)
            f.write(s+ "\n")

    wordIter.flush_log()
    wordIter._reset()

def write_tokens(N=None):
    textfile = "data/gibbon_daf_linked.txt"
    tokenfile = "data/gibbon_daf_tokens.txt"
    word_source = DAFWords(textfile=textfile)
    with open(tokenfile, 'w') as f:
        for i,word in enumerate(word_source.tokenize(N=N, lower=True)):
            f.write(word + "\n")
    print(f"Wrote {i+1} tokens to {tokenfile}.")
    
if __name__ == "__main__":

    write_tokens()
    