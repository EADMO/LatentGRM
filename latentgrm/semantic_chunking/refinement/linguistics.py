"""Soft boundary evidence on the original global BPE offsets.

Syntax is evidence, not a claim that punctuation implies independent meaning.
Code/math is masked with equal-length spaces for parsing, but lexed for costs.
"""
from bisect import bisect_left, bisect_right
from functools import lru_cache
import re

from ..structure import (_punctuation_boundaries, _CRITERION_RE,
                           _PROTECTED_ATOM_RE, split_original_cot)

SPECIAL = re.compile(r"```[\s\S]*?(?:```|\Z)|`[^`\n]*`|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|<[^<>\n]+>")
LEXEME = re.compile(r"\d+(?:[.,]\d+)+|[\w]+(?:['’][\w]+)*", re.UNICODE)
CONNECTIVE = re.compile(r"\b(?:because|although|unless|whereas|however|therefore|but|and|or|if|not|never|no)\s*$", re.I)
TIGHT = {"det", "amod", "compound", "nummod", "poss", "case", "prt", "ccomp", "xcomp",
         "advmod", "nsubj", "nsubjpass", "csubj", "csubjpass",
         "attr", "acomp", "oprd", "pobj", "pcomp"}
# spaCy attaches cc to the preceding predicate. Protecting that entire arc
# incorrectly penalizes the desirable cut BEFORE "but/and". The lexical
# CONNECTIVE rule still penalizes a cut immediately AFTER the connective.
STRONG = {"neg", "aux", "auxpass", "mark"}
CLAUSE = {"advcl", "relcl", "conj", "parataxis"}


def short_fragment_penalty(text, token_count):
    """Penalize incomplete template/subject fragments, not all short clauses.

    A response heading or a short complete verdict is a useful structural unit.
    A partial criterion heading or a bare subject is not the same thing.
    """
    if token_count > 5:
        return 0.0
    text = text.strip()
    if re.fullmatch(r'-\s*Criterion\s+\d+', text, re.I):
        return 8.0
    if re.fullmatch(r'(?:Not )?Met\.\s+(?:There|It|The(?: response)?)', text, re.I):
        return 8.0
    if re.fullmatch(r'Justification:\s*(?:The|A|An|This|That|Response)?', text, re.I):
        return 8.0
    subject = r'(?:The (?:response|answer)|Response [AB]|It|Both responses)'
    modifiers = r'(?:\s+(?:also|only|clearly|explicitly|directly))*'
    if re.fullmatch(r'(?:Justification:\s*)?' + subject + modifiers, text, re.I):
        return 8.0
    if re.fullmatch(subject + modifiers + r'\s+(?:is|are|was|were|has|have|does|provides|uses|mentions|contains|includes)', text, re.I):
        return 4.0
    return 0.0


def clause_span(token):
    """Only predicate-bearing clauses; restrictive relatives get no bonus."""
    if token.dep_ not in CLAUSE or token.pos_ not in {"VERB", "AUX"}:
        return None
    subtree = list(token.subtree)
    left, right = subtree[0], subtree[-1]
    if [t.i for t in subtree] != list(range(left.i, right.i + 1)):
        return None
    if token.dep_ == "relcl":
        previous = next((t for t in reversed(list(token.doc[:left.i])) if not t.is_space), None)
        if previous is None or previous.text != ",":
            return None
    return left.idx, right.idx + len(right)


@lru_cache(maxsize=2)
def get_parser(config):
    import spacy
    spacy.require_cpu()
    nlp = spacy.load(config.parser_model, exclude=["ner", "lemmatizer"])
    if "parser" not in nlp.pipe_names:
        raise RuntimeError("The pinned spaCy pipeline must contain a dependency parser")
    return nlp


def boundary_evidence(text, offsets, config):
    """Return arrays indexed by the BPE cut position (0..M), plus audit data."""
    count = len(offsets)
    ends = [end for _, end in offsets]
    # A boundary can represent any whitespace between adjacent token offsets.
    base = [2.0] * (count + 1)
    risk = [0] * (count + 1)
    intra = [False] * (count + 1)
    kinds = ["word"] * (count + 1)
    templates = [(m.start(), m.end()) for m in _PROTECTED_ATOM_RE.finditer(text)]
    headers = [(m.start(), m.end()) for m in _CRITERION_RE.finditer(text)]
    templates += headers
    try:
        regions = [(s.start, s.end) for s in split_original_cot(text)]
    except ValueError:
        regions = [(0, len(text))]
    region_starts = {left for left, _ in regions}

    def in_template(a, b):
        return any(a < right and b > left for left, right in templates)

    def mark_edge(char_end, value, kind, attach_punctuation=False):
        if attach_punctuation:
            while char_end < len(text) and text[char_end] in ',;:.!?)]}’”':
                char_end += 1
        if any(left < char_end < right for left, right in templates):
            return
        b = bisect_right(ends, char_end)
        if 0 < b < count and ends[b - 1] <= char_end:
            # Qwen may include leading whitespace in the following BPE token.
            # A clause start after that whitespace still maps to the cut just
            # before this token, as long as no lexical character is crossed.
            if not text[ends[b - 1]:char_end].strip() and value < base[b]:
                base[b], kinds[b] = value, kind

    for match in LEXEME.finditer(text):
        lo = bisect_right(ends, match.start())
        hi = bisect_left(ends, match.end())
        for index in range(max(1, lo + 1), min(count, hi + 1)):
            if offsets[index][0] < match.end() and ends[index - 1] > match.start():
                intra[index] = True
    # Overlapping byte-fallback offsets (e.g. a multi-byte Unicode character).
    for b in range(1, count):
        if ends[b - 1] > offsets[b][0]:
            intra[b] = True
        if CONNECTIVE.search(text[max(0, ends[b - 1] - 32):ends[b - 1]]):
            risk[b] = 2

    special = [(m.start(), m.end()) for m in SPECIAL.finditer(text)]
    def in_special(a, b):
        return any(a < right and b > left for left, right in special)

    for left, right in regions:
        for pos in _punctuation_boundaries(text, left, right, soft=False):
            mark_edge(pos, -config.sentence_reward, "sentence")
        for pos in _punctuation_boundaries(text, left, right, soft=True):
            mark_edge(pos, 1.0, "phrase")
    for _, right in headers:
        mark_edge(right, 0.5, "structure")
    # Long code/math remains splittable at lexical separators, without parsing
    # its contents as prose or protecting the entire long region.
    for left, right in special:
        for match in re.finditer(r"[,;=+*/]|\s+", text[left:right]):
            mark_edge(left + match.end(), 1.5, "lexical_separator")

    audit = {"parser": config.parser, "fallback": None, "masked_regions": len(special)}
    if config.parser == "spacy":
        nlp = get_parser(config)
        masked = list(text)
        for left, right in special + templates:
            masked[left:right] = ["\n" if ch == "\n" else " " for ch in text[left:right]]
        try:
            doc = nlp("".join(masked))
        except (ValueError, RuntimeError) as exc:
            audit["fallback"] = type(exc).__name__ + ": " + str(exc)[:160]
            doc = None
        if doc is not None:
            meaningful = [t for t in doc if t.is_alpha and t.pos_ in {"VERB", "AUX"}]
            if not doc.has_annotation("DEP") or not meaningful:
                audit["fallback"] = "no_dependency_or_predicate"
            else:
                audit["rule_only_sentences"] = 0
                # Sentence rewards come only from observed sentence punctuation.
                # A statistical parser must not invent sentence ends in rubric
                # headings (e.g. "- Criterion 5 | [Principle]:").
                for token in doc:
                    if token.is_space or in_special(token.idx, token.idx + len(token)) or in_template(token.idx, token.idx + len(token)):
                        continue
                    head = token.head
                    if head.is_space or in_template(head.idx, head.idx + len(head)):
                        continue
                    left = min(token.idx, head.idx)
                    right = max(token.idx + len(token), head.idx + len(head))
                    if any(left < start < right for start in region_starts):
                        continue
                    lo = bisect_right(ends, left)
                    hi = bisect_left(ends, right)
                    if token.dep_ in TIGHT | STRONG and hi - lo + 1 <= config.compact_span_tokens:
                        for b in range(max(1, lo + 1), min(count, hi + 1)):
                            risk[b] = max(risk[b], 2 if token.dep_ in STRONG else 1)
                    span = clause_span(token)
                    if span and not in_special(*span):
                        mark_edge(span[0], 0.5, "clause")
                        mark_edge(span[1], 0.5, "clause", attach_punctuation=True)
    # Keep closing punctuation on the left and opening delimiters on the right.
    # Apostrophes inside words are handled by the lexeme scanner, not as quotes.
    openings, closings = set(), set()
    for pattern in (r'"[^"\n]+"', r"(?<!\w)'[^'\n]+'(?!\w)", r'“[^”\n]+”', r'‘[^’\n]+’', r'`[^`\n]+`'):
        for match in re.finditer(pattern, text):
            openings.add(match.start())
            closings.add(match.end() - 1)
    compact_literals = set()
    for left, right in special:
        lo = bisect_right(ends, left)
        hi = bisect_left(ends, right)
        if hi - lo + 1 <= config.compact_span_tokens:
            compact_literals.update(range(max(1, lo + 1), min(count, hi + 1)))
    for b in range(1, count):
        pos = ends[b - 1]
        next_char = pos
        while next_char < len(text) and text[next_char].isspace():
            next_char += 1
        previous_char = pos - 1
        while previous_char >= 0 and text[previous_char].isspace():
            previous_char -= 1
        if any(left < pos < right for left, right in headers):
            base[b], kinds[b] = 6.0, "template_internal"
        # A determiner without its noun is not a useful prose cut. Exclude
        # uppercase A (the response identifier), literals, and ordinary words
        # merely ending in one of these character sequences.
        if not in_special(max(0, pos-1), pos) and re.search(r'\b(?:the|The|a|an|An)\s*$', text[max(0,pos-24):pos]):
            risk[b] = 2
        if b in compact_literals:
            base[b], kinds[b], risk[b] = 12.0, 'compact_literal', 2
        prose_closer = next_char < len(text) and text[next_char] in ',;:.!?)]}’”' and not in_special(next_char, next_char+1)
        prose_opener = previous_char >= 0 and text[previous_char] in '([{“‘' and not in_special(previous_char, previous_char+1)
        if prose_closer or next_char in closings or previous_char in openings or prose_opener:
            base[b], kinds[b], risk[b] = 8.0, "detached_punctuation", 2
        if kinds[b] == "sentence":
            risk[b] = 0
        if intra[b]:
            base[b], kinds[b] = 12.0, "intra_lexeme"
    audit["template_spans"] = len(templates)
    audit["quality_revision"] = "r6"
    costs = [base[b] + config.risk_weight * risk[b] for b in range(count + 1)]
    return costs, risk, intra, kinds, audit
