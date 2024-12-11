from dataclasses import dataclass, field
from typing import TypeVar
import re

initial_metadata = {
    'source': ['name', 'radiation', 'type', 'probe_particle', 'beam_size_name', 'beam_size', 'beam_shape', 'wavelength', 'wavelength_min', 'wavelength_max', 'wavelength_spread'],
    'detector': ['name', 'distance', 'offset', 'orientation', 'beam_center', 'pixel_size', 'slit_length'],
    'aperture': ['name', 'type', 'size_name', 'size', 'distance'],
    'collimation': ['name', 'lengths'],
    'process': ['name', 'date', 'description', 'term', 'notes'],
    'sample': ['name', 'sample_id', 'thickness', 'transmission', 'temperature', 'position', 'orientation', 'details'],
    'transmission_spectrum': ['name', 'timestamp', 'transmission', 'transmission_deviation'],
    'other': ['title', 'run', 'definition']
}

CASING_REGEX = r'[A-Z][a-z]*'

T = TypeVar('T')

@dataclass
class AsciiMetadataCategory[T]:
    values: dict[str, T] = field(default_factory=dict)

def default_categories() -> dict[str, AsciiMetadataCategory[str | int]]:
    return {key: AsciiMetadataCategory() for key in initial_metadata.keys()}

@dataclass
class AsciiReaderMetadata:
    # Key is the filename.
    filename_specific_metadata: dict[str, dict[str, AsciiMetadataCategory[str]]] = field(default_factory=dict)
    # True instead of str means use the casing to separate the filename.
    filename_separator: dict[str, str | bool] = field(default_factory=dict)
    master_metadata: dict[str, AsciiMetadataCategory[int]] = field(default_factory=default_categories)

    def filename_components(self, filename: str) -> list[str]:
        separator = self.filename_separator[filename]
        if isinstance(separator, str):
            splitted = re.split(f'{self.filename_separator[filename]}', filename)
        else:
            splitted = re.findall(CASING_REGEX, filename)
        # If the last component has a file extensions, remove it.
        last_component = splitted[-1]
        if '.' in last_component:
            pos = last_component.index('.')
            last_component = last_component[:pos]
            splitted[-1] = last_component
        return splitted

    def all_file_metadata(self, filename: str) -> dict[str, AsciiMetadataCategory[str]]:
        file_metadata = self.filename_specific_metadata[filename]
        components = self.filename_components(filename)
        # The ordering here is important. If there are conflicts, the second dictionary will override the first one.
        # Conflicts shouldn't really be happening anyway but if they do, we're gonna go with the master metadata taking
        # precedence for now.
        return_metadata: dict[str, AsciiMetadataCategory[str]] = {}
        for category_name, category in file_metadata.items():
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
        if isinstance(new_value, str):
            self.filename_specific_metadata[filename][category].values[key] = new_value
            # TODO: What about the master metadata? Until that's gone, that still takes precedence.
        elif isinstance(new_value, int):
            self.master_metadata[category].values[key] = new_value
        else:
            raise TypeError('Invalid type for new_value')

    def clear_metadata(self, category: str, key: str, filename: str):
        category_obj = self.filename_specific_metadata[filename][category]
        if key in category_obj.values:
            del category_obj.values[key]
        if key in self.master_metadata[category].values:
            del self.master_metadata[category].values[key]

    def add_file(self, new_filename: str):
        # TODO: Fix typing here. Pyright is showing errors.
        self.filename_specific_metadata[new_filename] = default_categories()
