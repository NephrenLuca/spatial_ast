"""
Post-hoc bracket balancer for decoded token sequences.

Ensures every ``(`` has a matching ``)`` and vice-versa.
Follows architecture.md Section 7.3.
"""

from __future__ import annotations

from typing import List

from core.tokenizer import TOKEN_LPAREN, TOKEN_RPAREN


class BracketBalancer:
    """
    Repairs bracket imbalance in a token sequence.

    - Unmatched ``)`` tokens are dropped.
    - Unclosed ``(`` tokens get a ``)`` appended at the end.
    """

    @staticmethod
    def check(tokens: List[int]) -> bool:
        """Return True if brackets are balanced."""
        depth = 0
        for tok in tokens:
            if tok == TOKEN_LPAREN:
                depth += 1
            elif tok == TOKEN_RPAREN:
                if depth == 0:
                    return False
                depth -= 1
        return depth == 0

    @staticmethod
    def repair(tokens: List[int]) -> List[int]:
        """
        Return a repaired copy of *tokens* with balanced brackets.
        """
        stack: List[int] = []
        repaired: List[int] = []
        for tok in tokens:
            if tok == TOKEN_LPAREN:
                stack.append(len(repaired))
                repaired.append(tok)
            elif tok == TOKEN_RPAREN:
                if stack:
                    stack.pop()
                    repaired.append(tok)
                # else: skip unmatched ")"
            else:
                repaired.append(tok)
        while stack:
            stack.pop()
            repaired.append(TOKEN_RPAREN)
        return repaired

    def check_and_repair(self, tokens: List[int]) -> List[int]:
        """Repair only if needed, otherwise return the original list."""
        if self.check(tokens):
            return tokens
        return self.repair(tokens)
