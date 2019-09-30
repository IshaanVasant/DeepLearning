. ./path.sh
set -x


echo """
i AY
like L AY K
computer K AH M P Y UW T ER
is IH Z
a AH
machine M AH SH IY N
learns L ER N Z
from F R AH M
human HH Y UW M AH N
learning L ER N IH NG
deep D IY P
 """ > dict
cat dict

sed "s/\s.*//g;s/([0-9])//g" dict  | sort | uniq > wordlist

echo """ 
i like computer
computer is a machine
machine learns from human
i am a human
i like machine 
i like machine learning 
machine learning is deep
deep learning
""" > text

echo """
<eps> 0
i 1
like 2
computer 3
is 4
a 5
machine 6
learns 7
from 8
human 9
learning 10
deep 11
#0 12
""" > words.syms

ngram-count -order 2 -text text -lm lm.2.arpa
cat lm.2.arpa


cat lm.2.arpa | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | \
    fstprint | \
    ./eps2disambig.pl |\
    ./s2eps.pl | \
    fstcompile --isymbols=words.syms \
      --osymbols=words.syms  \
      --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon > G.fst 

fstdraw  --isymbols=words.syms  --osymbols=words.syms  -portrait G.fst | dot -Tpdf > G.pdf


echo """<eps> 0
sil 1
AY 2
L 3
K 4
AH 5
M 6
P 7
Y 8
UW 9
T 10
ER 11
IH 12
Z 13
SH 14
IY 15
N 16
F 17
R 18
HH 19
NG 20
D 21
P 22
#0 23
""" > phones.syms


grep "#"  phones.syms | head -n1  | sed "s/.* //g" >  phones.disambig
grep "#"  words.syms | head -n1  | sed "s/.* //g" >  words.disambig

./make_lexicon_fst.pl dict 0.5 sil '#0' | \
   fstcompile --isymbols=phones.syms \
    --osymbols=words.syms \
    --keep_isymbols=false --keep_osymbols=false |\
   #fstaddselfloops  phones.disambig words.disambig  | \
   fstarcsort --sort_type=olabel \
   > L.fst

fstdraw  --isymbols=phones.syms  --osymbols=words.syms  -portrait L.fst | dot -Tpdf > L.pdf

