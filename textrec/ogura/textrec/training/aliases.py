"""Explicit, versioned character equivalence classes; never changes rendering."""
import json
import re
import unicodedata
from pathlib import Path


class CharacterAliases:
    def __init__(self, config=None):
        config = {"version": 1, "groups": []} if config is None else config
        if not isinstance(config, dict) or type(config.get("version")) is not int or config["version"] not in (1, 2) or not isinstance(config.get("groups"), list):
            raise ValueError("Expected character aliases version 1 or 2 and a groups list")
        # Human-readable notes are metadata, not part of checkpoint identity.
        if "_comment" in config:
            if not isinstance(config["_comment"], str):
                raise ValueError("Alias _comment must be a string")
            config = {key: value for key, value in config.items() if key != "_comment"}
        options = {"collapse_ascii_spaces", "compose_katakana_diacritics"}
        expected = {"version", "groups"} | (options if config["version"] == 2 else set())
        if set(config) != expected or any(type(config[k]) is not bool for k in options if k in config):
            raise ValueError("Invalid alias configuration keys or options")
        self.collapse_spaces = config.get("collapse_ascii_spaces", False)
        self.compose_katakana = config.get("compose_katakana_diacritics", False)
        self.mapping = {}
        groups = []
        for group in config["groups"]:
            if not isinstance(group, dict) or set(group) != {"representative", "members"}:
                raise ValueError("Each alias group requires representative and members")
            representative, members = group["representative"], group["members"]
            if not isinstance(members, list) or len(members) < 2 or any(not isinstance(c, str) or len(c) != 1 or (c.isspace() and c not in (' ', '\u3000')) or ord(c) < 32 for c in members):
                raise ValueError("Alias members must be at least two individual characters (only U+0020/U+3000 whitespace allowed)")
            if not isinstance(representative, str) or representative not in members:
                raise ValueError("Representative must belong to its group")
            if len(set(members)) != len(members) or set(members) & self.mapping.keys():
                raise ValueError("Alias members must be unique and groups disjoint")
            self.mapping.update(dict.fromkeys(members, representative))
            groups.append(dict(representative=representative, members=sorted(members)))
        for c in (' ', '\u3000'):
            if c in self.mapping and self.mapping[c] != ' ':
                raise ValueError('Spaces may only map to ASCII space')
        self.config = dict(version=config['version'], groups=sorted(groups, key=lambda g: g["representative"]))
        if config['version'] == 2:
            self.config.update(collapse_ascii_spaces=self.collapse_spaces,
                               compose_katakana_diacritics=self.compose_katakana)
        self.compositions = {}
        if self.compose_katakana:
            for cp in range(0x30a1, 0x30fb):
                for mark, combining in [('゛', '\u3099'), ('゜', '\u309a'), ('\u3099', '\u3099'), ('\u309a', '\u309a')]:
                    composed = unicodedata.normalize('NFC', chr(cp) + combining)
                    if len(composed) == 1:
                        self.compositions[chr(cp) + mark] = composed

    @classmethod
    def read(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))) if path else cls()

    def normalize(self, text):
        text = ''.join(self.mapping.get(c, c) for c in text)
        if self.compose_katakana:
            text = re.sub('[\u30a1-\u30fa][゛゜\u3099\u309a]',
                          lambda m: self.mapping.get(self.compositions.get(m[0], m[0]), self.compositions.get(m[0], m[0])), text)
        if self.collapse_spaces:
            text = re.sub(' +', ' ', text)
        return text
