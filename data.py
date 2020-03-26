import requests
from bs4 import BeautifulSoup

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

    def __init__(self, fname="data/gibbon_daf.html"):
        self.fname = fname
        self.parsed = None
        self.load_parsed()

        self._current_body_par = None

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
                yield par.text
            else:
                for el in par.contents:
                    if el.name is None:
                        #this is plain text
                        yield str(el)
                    elif el.name == "a" and "href" in el.attrs:
                        id = el["href"].lstrip('#')
                        try:
                            foot_par = self._get_footnote_par(id)
                        except NoFootnoteError:
                            print(f"Could not find footnote for {id}, skipping.")
                        yield foot_par.text


if __name__ == "__main__":

    wordIter = DAFIterator()
    outfile = "data/gibbon_daf_linked.txt"
    with open(outfile, 'w') as f:
        for i,s in enumerate(wordIter.linked_text_paragraphs()):
            if i%1000 == 0:
                print(i)
            f.write(s+ "\n")
