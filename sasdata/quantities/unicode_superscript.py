
_ascii_version = "0123456789-"
_unicode_version = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"

def int_as_unicode_superscript(number: int):
    string = str(number)

    for old, new in zip(_ascii_version, _unicode_version):
        string = string.replace(old, new)

    return string

