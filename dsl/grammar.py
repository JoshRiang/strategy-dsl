"""DSL grammar reference (informal documentation).

This module does not parse anything — it documents the grammar that the
hand-written recursive-descent parser in :mod:`dsl.parser` accepts. Keeping
the rules here lets users (and tests) reference a single source of truth.

BNF-like summary
================

::

    program        := statement*
    statement      := assignment | when_expr
    assignment     := IDENT '=' expression NEWLINE
    when_expr      := 'WHEN' expression 'THEN' expression ('ELSE' expression)? NEWLINE

    expression     := comparison
    comparison     := sum ( ( '>' | '<' | '>=' | '<=' | '==' | '!=' ) sum )*
    sum            := product ( ( '+' | '-' ) product )*
    product        := unary ( ( '*' | '/' ) unary )*
    unary          := ( '-' | '+' )? primary
    primary        := NUMBER | STRING | IDENT call?
    call           := '(' ( expression ( ',' expression )* )? ')'

Lexical tokens
==============

* ``IDENT``     — ``[A-Za-z_][A-Za-z0-9_]*``
* ``NUMBER``    — ``[0-9]+ ( '.' [0-9]+ )?``
* ``STRING``    — ``'"' ... '"'``
* ``NEWLINE``   — ``\\n`` or ``;``
* ``COMMENT``   — ``# ...`` until end of line

Signals and builtins
====================

Signals (case-insensitive) are exposed by the compiler via the price
DataFrame columns. The DSL reserves the following builtins (see
:mod:`dsl.stdlib`):

``sma, ema, rank, ts_rank, zscore, std, mean, returns, sign, abs, log,
clip, scale, neutralize``
"""

# Reserved keywords (lowercased). Used by the tokenizer/parser to distinguish
# control tokens from identifiers.
KEYWORDS = {
    "when",
    "then",
    "else",
    "and",
    "or",
    "not",
    "true",
    "false",
}

# Built-in function names (lowercased). These are dispatched by the compiler
# to the implementations in ``dsl.stdlib``.
BUILTINS = {
    "sma",
    "ema",
    "rank",
    "ts_rank",
    "zscore",
    "std",
    "mean",
    "returns",
    "sign",
    "abs",
    "log",
    "clip",
    "scale",
    "neutralize",
}

# Reserved signal names — these refer to columns of the input price
# DataFrame rather than to user-defined variables.
SIGNALS = {"open", "high", "low", "close", "volume"}

# Operator precedence table. Lower index = tighter binding.
PRECEDENCE = [
    ("or",),
    ("and",),
    ("not",),
    ("==", "!="),
    (">", "<", ">=", "<="),
    ("+", "-"),
    ("*", "/"),
    ("u_neg", "u_pos"),  # unary
]
