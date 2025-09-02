from typing import Union, List, Dict, Optional
from general.structures.data_batch import DataBatch
import re

class InvalidCharacterChecker:
    """
    Validates text data for presence of invalid or unexpected characters.

    This class checks input text for characters outside a specified valid set.
    It can be configured with custom character sets and handles both single strings
    and batched data inputs. Designed for integration with data validation pipelines.
    """

    def __init__(self, valid_charset: set=None, name: str=None):
        """
        Initialize the invalid character checker.

        Args:
            valid_charset (set, optional): Set of valid characters. Defaults to ASCII printable characters.
            name (str, optional): Name identifier for the validator.
        """
        import string
        self.valid_charset = valid_charset or set(string.printable)
        self.name = name or self.__class__.__name__

    def check_text(self, text: Union[str, List[str]]) -> dict:
        """
        Check text for invalid characters.

        Args:
            text (Union[str, List[str]]): Text or list of texts to check.

        Returns:
            dict: Report containing invalid characters found, their positions, and counts.
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        invalid_chars = {}
        total_invalid_count = 0
        for (text_idx, txt) in enumerate(texts):
            for (pos, char) in enumerate(txt):
                if char not in self.valid_charset:
                    total_invalid_count += 1
                    if char not in invalid_chars:
                        invalid_chars[char] = {'count': 0, 'positions': []}
                    invalid_chars[char]['count'] += 1
                    invalid_chars[char]['positions'].append((text_idx, pos))
        return {'valid': total_invalid_count == 0, 'invalid_characters': invalid_chars, 'total_invalid_count': total_invalid_count}

    def validate(self, data: DataBatch) -> bool:
        """
        Validate a DataBatch for text character validity.

        Args:
            data (DataBatch): Batch of data to validate.

        Returns:
            bool: True if all text passes validation, False otherwise.
        """
        if isinstance(data.data, list):
            if len(data.data) > 0:
                if isinstance(data.data[0], (list, tuple)):
                    for row in data.data:
                        for item in row:
                            if isinstance(item, str):
                                result = self.check_text(item)
                                if not result['valid']:
                                    return False
                elif isinstance(data.data[0], dict):
                    for record in data.data:
                        for value in record.values():
                            if isinstance(value, str):
                                result = self.check_text(value)
                                if not result['valid']:
                                    return False
                else:
                    for item in data.data:
                        if isinstance(item, str):
                            result = self.check_text(item)
                            if not result['valid']:
                                return False
        elif hasattr(data.data, 'shape') and len(data.data.shape) == 2:
            import numpy as np
            for row in data.data:
                for item in row:
                    if isinstance(item, str):
                        result = self.check_text(item)
                        if not result['valid']:
                            return False
        elif isinstance(data.data, str):
            result = self.check_text(data.data)
            return result['valid']
        return True

def expand_contractions(text: Union[str, List[str]], contraction_mapping: dict=None) -> Union[str, List[str]]:
    """
    Expand common English contractions in text to their full forms (e.g., "don't" -> "do not").

    This function processes input text to replace contracted forms with their expanded equivalents.
    It supports both single strings and lists of strings. A default mapping of common contractions
    is used unless a custom mapping is provided.

    Args:
        text (Union[str, List[str]]): The input text or list of texts to process.
        contraction_mapping (dict, optional): A dictionary mapping contractions to their expansions.
                                              If None, a predefined set of common contractions is used.

    Returns:
        Union[str, List[str]]: The processed text with contractions expanded. The return type matches the input type.

    Examples:
        >>> expand_contractions("I don't like it.")
        "I do not like it."

        >>> expand_contractions(["It's great!", "We're happy."])
        ["It is great!", "We are happy."]
    """
    default_contractions = {"ain't": 'am not', "aren't": 'are not', "can't": 'cannot', "could've": 'could have', "couldn't": 'could not', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'll": 'he will', "he's": 'he is', "how'd": 'how did', "how'll": 'how will', "how's": 'how is', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it would', "it'll": 'it will', "it's": 'it is', "let's": 'let us', "might've": 'might have', "mightn't": 'might not', "must've": 'must have', "mustn't": 'must not', "shan't": 'shall not', "she'd": 'she would', "she'll": 'she will', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "that'd": 'that would', "that's": 'that is', "there'd": 'there would', "there's": 'there is', "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have', "wasn't": 'was not', "we'd": 'we would', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what're": 'what are', "what's": 'what is', "what've": 'what have', "where'd": 'where did', "where's": 'where is', "who'll": 'who will', "who's": 'who is', "won't": 'will not', "would've": 'would have', "wouldn't": 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', "you've": 'you have'}
    mapping = contraction_mapping if contraction_mapping is not None else default_contractions
    sorted_contractions = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile('\\b(' + '|'.join((re.escape(c) for c in sorted_contractions)) + ')\\b', re.IGNORECASE)

    def expand_single_text(txt: str) -> str:
        """Expand contractions in a single text string."""

        def replace_match(match):
            contraction = match.group(0)
            expansion = mapping.get(contraction.lower(), contraction)
            if contraction.isupper():
                expansion = expansion.upper()
            elif contraction[0].isupper():
                expansion = expansion.capitalize()
            return expansion
        return pattern.sub(replace_match, txt)
    if isinstance(text, str):
        return expand_single_text(text)
    elif isinstance(text, list):
        return [expand_single_text(t) for t in text]
    else:
        raise TypeError('Input must be a string or list of strings')

def convert_to_title_case(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Convert input text to title case (capitalizing the first letter of each word).

    This function transforms text so that the first character of each word is capitalized
    and the rest are lowercase. It supports both single strings and lists of strings.

    Args:
        text (Union[str, List[str]]): The input text or list of texts to convert.

    Returns:
        Union[str, List[str]]: The processed text in title case. The return type matches the input type.

    Examples:
        >>> convert_to_title_case("hello world")
        "Hello World"

        >>> convert_to_title_case(["this is ONE", "ANOTHER test"])
        ["This Is One", "Another Test"]
    """
    if isinstance(text, str):
        return text.title()
    elif isinstance(text, list):
        return [t.title() for t in text]
    else:
        raise TypeError('Input must be a string or list of strings')

def convert_to_uppercase(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Convert input text to uppercase.

    This function transforms all alphabetic characters in the input text to uppercase.
    It supports both single strings and lists of strings for batch processing.

    Args:
        text (Union[str, List[str]]): The input text or list of texts to convert.

    Returns:
        Union[str, List[str]]: The processed text in uppercase. The return type matches the input type.

    Examples:
        >>> convert_to_uppercase("hello world")
        "HELLO WORLD"

        >>> convert_to_uppercase(["lowercase", "MiXeD CaSe"])
        ["LOWERCASE", "MIXED CASE"]
    """
    if isinstance(text, str):
        return text.upper()
    elif isinstance(text, list):
        return [t.upper() for t in text]
    else:
        raise TypeError('Input must be a string or list of strings')

def remove_special_characters(text: Union[str, List[str]], keep_chars: Optional[str]=None) -> Union[str, List[str]]:
    """
    Remove special characters from text, retaining only alphanumeric characters and specified exceptions.

    This function strips out non-alphanumeric characters from input text, with optional
    exceptions for specific characters (e.g., spaces, underscores). It supports both
    single strings and lists of strings for flexible usage.

    Args:
        text (Union[str, List[str]]): The input text or list of texts to process.
        keep_chars (str, optional): A string of characters to retain (e.g., " _-"). 
                                    By default, only alphanumeric characters are kept.

    Returns:
        Union[str, List[str]]: The processed text with special characters removed. 
                               The return type matches the input type.

    Examples:
        >>> remove_special_characters("hello, world!")
        "helloworld"

        >>> remove_special_characters(["email@domain.com", "test-case"], keep_chars="-")
        ["emaildomaincom", "test-case"]
    """
    import re
    escaped_keep_chars = re.escape(keep_chars) if keep_chars else ''
    pattern = f'[^a-zA-Z0-9{escaped_keep_chars}]'

    def clean_single_text(t: str) -> str:
        return re.sub(pattern, '', t)
    if isinstance(text, str):
        return clean_single_text(text)
    elif isinstance(text, list):
        return [clean_single_text(t) for t in text]
    else:
        raise TypeError('Input must be a string or list of strings')

def correct_typos_with_dictionary(text: Union[str, List[str]], typo_dict: Optional[Dict[str, str]]=None) -> Union[str, List[str]]:
    """
    Correct common typos in text using a predefined or user-provided dictionary.

    This function replaces known misspellings with their correct forms based on a lookup dictionary.
    It supports both single strings and lists of strings. If no dictionary is provided, a default
    set of common typos is used.

    Args:
        text (Union[str, List[str]]): The input text or list of texts to process.
        typo_dict (Dict[str, str], optional): A dictionary mapping typos to corrections.
                                              If None, a built-in dictionary of common typos is used.

    Returns:
        Union[str, List[str]]: The processed text with typos corrected. The return type matches the input type.

    Examples:
        >>> correct_typos_with_dictionary("I havv a drem")
        "I have a dream"

        >>> custom_dict = {"teh": "the", "recieve": "receive"}
        >>> correct_typos_with_dictionary(["teh quick brown fox", "recieve updates"], typo_dict=custom_dict)
        ["the quick brown fox", "receive updates"]
    """
    default_typo_dict = {'havv': 'have', 'drem': 'dream', 'teh': 'the', 'recieve': 'receive', 'occured': 'occurred', 'seperate': 'separate', 'definately': 'definitely', 'accomodate': 'accommodate', 'neccessary': 'necessary', 'begining': 'beginning', 'existance': 'existence', 'maintainance': 'maintenance', 'occuring': 'occurring', 'refered': 'referred', 'relize': 'realize', 'succesful': 'successful', 'writen': 'written', 'ths': 'this', 'gret': 'great'}
    mapping = typo_dict if typo_dict is not None else default_typo_dict
    if not mapping:
        if isinstance(text, str):
            return text
        elif isinstance(text, list):
            return text[:]
        else:
            raise TypeError('Input must be a string or list of strings')
    sorted_typos = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile('\\b(' + '|'.join((re.escape(typo) for typo in sorted_typos)) + ')\\b', re.IGNORECASE)

    def correct_single_text(txt: str) -> str:
        """Correct typos in a single text string."""

        def replace_match(match):
            typo = match.group(0)
            correction = mapping.get(typo.lower(), typo)
            if not correction:
                return correction
            if typo.isupper():
                return correction.upper()
            elif typo and typo[0].isupper():
                return correction.capitalize()
            else:
                return correction
        return pattern.sub(replace_match, txt)
    if isinstance(text, str):
        return correct_single_text(text)
    elif isinstance(text, list):
        return [correct_single_text(t) for t in text]
    else:
        raise TypeError('Input must be a string or list of strings')