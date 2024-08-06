"""

Post processing for loaded files

"""

def fix_mantid_units_error(data: SasData) -> SasData:
    pass



def apply_fixes(data: SasData, mantid_unit_error=True):
    if mantid_unit_error:
        data = fix_mantid_units_error(data)

    return data
