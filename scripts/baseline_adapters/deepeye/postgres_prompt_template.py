"""External dialect-only adaptation of native BIRD/Spider stage instructions."""

_DATE_RULE = "Utilize `STRFTIME()` for date manipulation (e.g., `STRFTIME('%Y', SOMETIME)` to extract the year)."
_FK_RULE = 'If there are multiple tables to JOIN, you MUST ensure that the joined tables have EXPLICIT FOREIGN KEYS between them. For example, "TableA -> TableB, TableC -> TableB", directly join TableA and TableC is NOT ALLOWED, you must join TableA and TableB, and then join TableB and TableC.'


def postgres_template(template: str) -> str:
    """Adapt templates before interpolation so question/evidence stay verbatim."""
    template = template.replace("SQLite", "PostgreSQL")
    template = template.replace(_DATE_RULE, "Use PostgreSQL EXTRACT, DATE_TRUNC and interval arithmetic for dates; for example EXTRACT(YEAR FROM SOMETIME).")
    template = template.replace(_FK_RULE, "The supplied Meta does not declare foreign keys. Use relationships justified by the question, evidence and column descriptions; an absent declaration does not prohibit a JOIN. Do not invent relationships.")
    template = template.replace("(`table_name` or `column_name`)", '("table_name" or "column_name")')
    return (
        "# Database dialect: PostgreSQL\n"
        "Use only the supplied public tables and columns. Return one read-only query. "
        "JSONB Fields in a column description describe keys inside that JSONB column, not additional physical columns. "
        "Use ->, ->>, #> or #>> as appropriate and cast extracted text before arithmetic. "
        "Use double quotes for SQL identifiers and single quotes for string literals.\n"
        + template
    )
