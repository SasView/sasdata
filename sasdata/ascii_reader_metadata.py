import re
from dataclasses import dataclass, field
from typing import TypeVar

initial_metadata = {
    'source': ['name', 'radiation', 'type', 'probe_particle', 'beam_size_name', 'beam_size', 'beam_shape', 'wavelength', 'wavelength_min', 'wavelength_max', 'wavelength_spread'],
    'detector': ['name', 'distance', 'offset', 'orientation', 'beam_center', 'pixel_size', 'slit_length'],
    'aperture': ['name', 'type', 'size_name', 'size', 'distance'],
    'collimation': ['name', 'lengths'],
    'process': ['name', 'date', 'description', 'term', 'notes'],
    'sample': ['name', 'sample_id', 'thickness', 'transmission', 'temperature', 'position', 'orientation', 'details'],
    'transmission_spectrum': ['name', 'timestamp', 'transmission', 'transmission_deviation'],
    'magnetic': ['demagnetizing_field', 'saturation_magnetization', 'applied_magnetic_field', 'counting_index'],
    'other': ['title', 'run', 'definition']
}

CASING_REGEX = r'[A-Z][a-z]*'

# First item has the highest precedence.
SEPARATOR_PRECEDENCE = [
    '_',
    '-',
]
# If none of these characters exist in that string, use casing. See init_separator

T = TypeVar('T')

# TODO: There may be a better place for this.
pairings = {'I': 'dI', 'Q': 'dQ', 'Qx': 'dQx', 'Qy': 'dQy', 'Qz': 'dQz'}
pairing_error = {value: key for key, value in pairings.items()}
# Allows this to be bidirectional.
bidirectional_pairings = pairings | pairing_error

@dataclass
class AsciiMetadataCategory[T]:
    values: dict[str, T] = field(default_factory=dict)

def default_categories() -> dict[str, AsciiMetadataCategory[str | int]]:
    return {key: AsciiMetadataCategory() for key in initial_metadata}

@dataclass
class AsciiReaderMetadata:
    # Key is the filename.
    filename_specific_metadata: dict[str, dict[str, AsciiMetadataCategory[str]]] = field(default_factory=dict)
    # True instead of str means use the casing to separate the filename.
    filename_separator: dict[str, str | bool] = field(default_factory=dict)
    master_metadata: dict[str, AsciiMetadataCategory[int]] = field(default_factory=default_categories)

    def init_separator(self, filename: str):
        separator = next(filter(lambda c: c in SEPARATOR_PRECEDENCE, filename), True)
        self.filename_separator[filename] = separator

    def filename_components(self, filename: str, cut_off_extension: bool = True, capture: bool = False) -> list[str]:
        """Split the filename into several components based on the current separator for that file."""
        separator = self.filename_separator[filename]
        # FIXME: This sort of string construction may be an issue. Might need an alternative.
        base_str = '({})' if capture else '{}'
        if isinstance(separator, str):
            splitted = re.split(base_str.replace('{}', separator), filename)
        else:
            splitted = re.findall(base_str.replace('{}', CASING_REGEX), filename)
        # If the last component has a file extensions, remove it.
        last_component = splitted[-1]
        if cut_off_extension and '.' in last_component:
            pos = last_component.index('.')
            last_component = last_component[:pos]
            splitted[-1] = last_component
        return splitted

    def purge_unreachable(self, filename: str):
        """This is used when the separator has changed. If lets say we now have 2 components when there were 5 but the
        3rd component was selected, this will now produce an index out of range exception. Thus we'll need to purge this
        to stop exceptions from happening."""
        components = self.filename_components(filename)
        component_length = len(components)
        # Converting to list as this mutates the dictionary as it goes through it.
        for category_name, category in list(self.master_metadata.items()):
            for key, value in list(category.values.items()):
                if value >= component_length:
                    del self.master_metadata[category_name].values[key]

    def all_file_metadata(self, filename: str) -> dict[str, AsciiMetadataCategory[str]]:
        """Return all of the metadata for known for the specified filename. This
        will combin the master metadata specified for all files with the
        metadata specific to that filename."""
        file_metadata = self.filename_specific_metadata[filename]
        components = self.filename_components(filename)
        # The ordering here is important. If there are conflicts, the second dictionary will override the first one.
        # Conflicts shouldn't really be happening anyway but if they do, we're gonna go with the master metadata taking
        # precedence for now.
        return_metadata: dict[str, AsciiMetadataCategory[str]] = {}
        for category_name, category in (file_metadata | self.master_metadata).items():
            combined_category_dict = category.values | self.master_metadata[category_name].values
            new_category_dict: dict[str, str] = {}
            for key, value in combined_category_dict.items():
                if isinstance(value, str):
                    new_category_dict[key] = value
                elif isinstance(value, int):
                    new_category_dict[key] = components[value]
                else:
                    raise TypeError(f'Invalid value for {key} in {category_name}')
            new_category = AsciiMetadataCategory(new_category_dict)
            return_metadata[category_name] = new_category
        return return_metadata
    def get_metadata(self, category: str, value: str, filename: str, error_on_not_found=False) -> str | None:
        """Get a particular piece of metadata for the filename."""
        components = self.filename_components(filename)

        # We prioritise the master metadata.

        # TODO: Assumes category in master_metadata exists. Is this a reasonable assumption? May need to make sure it is
        # definitely in the dictionary.
        if value in self.master_metadata[category].values:
            index = self.master_metadata[category].values[value]
            return components[index]
        target_category = self.filename_specific_metadata[filename][category].values
        if value in target_category:
            return target_category[value]
        if error_on_not_found:
            raise ValueError('value does not exist in metadata.')
        else:
            return None

    def update_metadata(self, category: str, key: str, filename: str, new_value: str | int):
        """Update the metadata for a filename. If the new_value is a string,
        then this new metadata will be specific to that file. Otherwise, if
        new_value is an integer, then this will represent the component of the
        filename that this metadata applies to all."""
        if isinstance(new_value, str):
            self.filename_specific_metadata[filename][category].values[key] = new_value
            # TODO: What about the master metadata? Until that's gone, that still takes precedence.
        elif isinstance(new_value, int):
            self.master_metadata[category].values[key] = new_value
        else:
            raise TypeError('Invalid type for new_value')

    def clear_metadata(self, category: str, key: str, filename: str):
        """Remove any metadata recorded for a certain filename."""
        category_obj = self.filename_specific_metadata[filename][category]
        if key in category_obj.values:
            del category_obj.values[key]
        if key in self.master_metadata[category].values:
            del self.master_metadata[category].values[key]

    def add_file(self, new_filename: str):
        """Add a filename to the metadata, filling it with some default
        categories."""
        # TODO: Fix typing here. Pyright is showing errors.
        self.filename_specific_metadata[new_filename] = default_categories()
