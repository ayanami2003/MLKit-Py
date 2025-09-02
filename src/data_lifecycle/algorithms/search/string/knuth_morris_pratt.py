from typing import List

def knuth_morris_pratt_search(text: str, pattern: str) -> List[int]:
    """
    Perform pattern matching in a text string using the Knuth-Morris-Pratt (KMP) algorithm.
    
    This function searches for all occurrences of a given pattern within a text string
    using the KMP algorithm, which preprocesses the pattern to avoid unnecessary character
    comparisons during the search process.
    
    Args:
        text (str): The text string to search within.
        pattern (str): The pattern string to search for.
        
    Returns:
        List[int]: A list of starting indices where the pattern is found in the text.
                  Returns an empty list if no matches are found.
                  
    Examples:
        >>> knuth_morris_pratt_search("ABABDABACDABABCABCABCABCABC", "ABABCABCABC")
        [10, 14]
        
        >>> knuth_morris_pratt_search("hello world", "world")
        [6]
        
        >>> knuth_morris_pratt_search("aaaaaa", "aa")
        [0, 1, 2, 3, 4]
    """
    if not pattern:
        return list(range(len(text) + 1))
    if not text:
        return []
    if len(pattern) > len(text):
        return []

    def _compute_lps_array(pattern: str) -> List[int]:
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
        return lps
    lps = _compute_lps_array(pattern)
    matches = []
    i = 0
    j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            matches.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return matches