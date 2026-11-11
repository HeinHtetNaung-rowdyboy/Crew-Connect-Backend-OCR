
"""Conservative deterministic extractor: absent/unclear fields are null."""

import re
from collections import Counter

from app.extraction.date_parser import first_date, DATE_PATTERN


# --------------------------------------------------
# DATE LABELS
# --------------------------------------------------

DATE_LABELS = {
    "training_date": [
        r"date\s+of\s+training",
        r"training\s+date",
        r"conducted\s+(?:by|on|at).*?on\s+the\s+date\s+of",
        r"on\s+the\s+date\s+of",
        r"date\s+of\s+completion",
    ],
    "issue_date": [
        r"date\s+of\s+issue",
        r"issued\s+on",
        r"issue\s+date",
    ],
    "examination_date": [
        r"date\s+of\s+examination",
        r"examination\s+date",
        r"exam\s+date",
    ],
    "expire_date": [
        r"date\s+of\s+expiry",
        r"date\s+of\s+expiration",
        r"expir(?:y|ation)\s+date",
        r"valid\s+until",
        r"valid\s+to",
        r"date\s+of\s+expir(?:y|ation)",
    ],
}


# --------------------------------------------------
# CERTIFICATE NUMBER PATTERNS
# --------------------------------------------------

NUMBER_PATTERNS = [
    r"(?:certificate\s*(?:number|no\.?|#)|"
    r"license\s*(?:number|no\.?|#)|"
    r"passport\s*(?:number|no\.?|#))"
    r"\s*[:#-]?\s*"
    r"([A-Z0-9][A-Z0-9 /-]{3,35})",
]


# --------------------------------------------------
# COVID VACCINE: FIRST DOSE DATE
# --------------------------------------------------

def extract_covid_first_dose(text: str) -> str | None:
    """
    Extract the vaccination date associated with
    the 1st Dose / First Dose row.

    Only use a date when the association is
    sufficiently clear.

    Return normalized YYYY-MM-DD or None.
    """

    dose_pattern = re.compile(
        r"\b(?:1st|first)\s*dose\b",
        re.IGNORECASE
    )

    next_dose_pattern = re.compile(
        r"\b(?:2nd|second|3rd|third)\s*dose\b"
        r"|\bbooster\b"
        r"|\bothers\b",
        re.IGNORECASE
    )

    lines = text.splitlines()

    # First try dates on the same line as 1st Dose.
    for line in lines:
        if dose_pattern.search(line):
            dates = [
                first_date(match.group(0))
                for match in DATE_PATTERN.finditer(line)
            ]

            dates = [date for date in dates if date]

            if len(set(dates)) == 1:
                return dates[0]

    # Handle OCR output where a dose row is split
    # across multiple adjacent lines.
    for index, line in enumerate(lines):

        match = dose_pattern.search(line)

        if not match:
            continue

        nearby = [line[match.end():]]

        # Inspect a small number of following lines.
        for following in lines[index + 1:index + 4]:

            if next_dose_pattern.search(following):
                break

            nearby.append(following)

        candidate = " ".join(nearby)

        dates = [
            first_date(match.group(0))
            for match in DATE_PATTERN.finditer(candidate)
        ]

        dates = [date for date in dates if date]

        # Avoid guessing when multiple dates appear.
        if len(set(dates)) == 1:
            return dates[0]

    return None


# --------------------------------------------------
# SEAMAN BOOK: BOOK NUMBER
# --------------------------------------------------

def extract_seaman_book_number(text: str) -> str | None:
    """
    Extract Book No. or Book Number from
    a Seaman Book certificate.

    Avoid unrelated Passport No. and SID No.
    """

    pattern = re.compile(
        r"\b(?:"
        r"seaman'?s?\s+book\s*(?:no\.?|number)"
        r"|seafarer'?s?\s+book\s*(?:no\.?|number)"
        r"|book\s*(?:no\.?|number)"
        r")"
        r"\s*[:#-]?\s*"
        r"(?P<number>[A-Z0-9][A-Z0-9/-]{3,29})\b",
        re.IGNORECASE
    )

    for match in pattern.finditer(text):

        number = match.group("number").strip()

        if number.lower() in (
            "date",
            "name",
            "number",
            "passport",
            "issue",
            "expiry",
        ):
            continue

        return number

    return None


# --------------------------------------------------
# GENERIC FIELD EXTRACTION
# --------------------------------------------------

def extract_fields(
    text: str,
    certificate_name: str = ""
) -> dict:
    """
    Extract date and certificate-number fields.

    Also apply certificate-specific extraction for:
      - COVID Vaccine
      - Seaman Book (MM)
      - PSSR
      - All in One
    """

    output = {
        key: None
        for key in (*DATE_LABELS, "license_number")
    }

    # ----------------------------------------------
    # 1. Generic date extraction
    # ----------------------------------------------

    for field, labels in DATE_LABELS.items():

        for label in labels:

            for match in re.finditer(
                label,
                text,
                flags=re.IGNORECASE
            ):

                nearby = text[
                    match.end():match.end() + 110
                ].split("\n", 2)[:2]

                candidate = " ".join(nearby)

                # Avoid borrowing a date from a
                # neighboring labeled field.
                candidate = re.split(
                    r"\b(?:"
                    r"date of birth|"
                    r"date of issue|"
                    r"date of expiry|"
                    r"certificate no|"
                    r"passport no"
                    r")\b",
                    candidate,
                    flags=re.IGNORECASE
                )[0]

                date = first_date(candidate)

                if date:
                    output[field] = date
                    break

            if output[field]:
                break

    # ----------------------------------------------
    # 2. PSSR-specific training date
    # ----------------------------------------------

    if (
        certificate_name == "PSSR"
        and not output["training_date"]
    ):

        match = re.search(
            r"on\s+the\s+date\s+of\s*\(?\s*"
            r"(" + DATE_PATTERN.pattern + r")",
            text,
            re.IGNORECASE
        )

        if match:
            output["training_date"] = first_date(
                match.group(0)
            )

    # ----------------------------------------------
    # 3. Generic certificate/license number
    # ----------------------------------------------

    for pattern in NUMBER_PATTERNS:

        for match in re.finditer(
            pattern,
            text,
            re.IGNORECASE
        ):

            number = match.group(1).strip(" -:/")

            number = re.split(
                r"\s{3,}|\n|"
                r"\b(?:date|sirb|s[il]rb|birth)\b",
                number,
                flags=re.IGNORECASE
            )[0].strip()

            if (
                number
                and not number.lower().startswith(
                    ("date", "no ")
                )
            ):

                output["license_number"] = number
                break

        if output["license_number"]:
            break

    # ----------------------------------------------
    # 4. PSSR-specific certificate number
    # ----------------------------------------------

    if (
        certificate_name == "PSSR"
        and not output["license_number"]
    ):

        match = re.search(
            r"\bMES\s*/\s*HSB\s*"
            r"\d{1,3}\s*[-–]\s*"
            r"\d{1,4}\s*/\s*20\d{2}\b",
            text,
            re.IGNORECASE
        )

        if match:
            output["license_number"] = re.sub(
                r"\s+",
                " ",
                match.group(0).strip()
            )

    # ----------------------------------------------
    # 5. All in One-specific expiry date
    # ----------------------------------------------

    if (
        certificate_name == "All in One"
        and not output["expire_date"]
    ):

        if re.search(
            r"date\s+of\s+expir|date ofexpiry",
            text,
            re.IGNORECASE
        ):

            dates = [
                first_date(match.group(0))
                for match in DATE_PATTERN.finditer(text)
            ]

            dates = [
                date
                for date in dates
                if date and (
                    not output["issue_date"]
                    or date > output["issue_date"]
                )
            ]

            counts = Counter(dates)

            if counts:

                date, frequency = counts.most_common(1)[0]

                if frequency >= 2:
                    output["expire_date"] = date

    # ----------------------------------------------
    # 6. COVID Vaccine-specific first dose
    # ----------------------------------------------

    if certificate_name == "COVID Vaccine":

        first_dose_date = extract_covid_first_dose(
            text
        )

        # This is a business-rule mapping:
        # first-dose vaccination date -> training_date
        # -> start_date in validator.py.
        #
        # Do not substitute a booster or second-dose
        # date if the first-dose date is unclear.
        output["training_date"] = first_dose_date

    # ----------------------------------------------
    # 7. Seaman Book-specific Book No.
    # ----------------------------------------------

    if certificate_name in (
        "선원수첩/Seaman Book (MM)",
        "Seaman Book (MM)",
        "Seaman Book",
    ):

        book_number = extract_seaman_book_number(
            text
        )

        if book_number:
            output["license_number"] = book_number

    return output
