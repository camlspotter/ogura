"""Reject overlapping source spans and repeated long text fragments."""
from bisect import bisect_left, insort
from collections import defaultdict
import hashlib

VERSION = 2
SHINGLE_LENGTH = 16


def fingerprints(text):
    return {hashlib.blake2b(text[i:i+SHINGLE_LENGTH].encode(),digest_size=8).digest()
            for i in range(max(1,len(text)-SHINGLE_LENGTH+1))}


class DiversityFilter:
    def __init__(self):
        self.intervals=defaultdict(list)
        self.fragments=set()

    def overlaps(self,article_id,start,end):
        spans=self.intervals.get(article_id,[])
        i=bisect_left(spans,(start,end))
        return (i>0 and spans[i-1][1]>start) or (i<len(spans) and spans[i][0]<end)

    def allows(self,article_id,start,end,text):
        return not self.overlaps(article_id,start,end) and self.fragments.isdisjoint(fingerprints(text))

    def add(self,article_id,start,end,text):
        insort(self.intervals[article_id],(start,end))
        self.fragments.update(fingerprints(text))
