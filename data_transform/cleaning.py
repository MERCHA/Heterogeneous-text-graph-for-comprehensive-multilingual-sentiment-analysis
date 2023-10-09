"""
This module hold all the functions required to perform data cleaning
"""

import re


def remove_duplicated_text(data):
    """
    Remove repeated text and keep just the first

    Parameters
    ----------
        data : dataframe
            The data to be processed, the data must contain text column

    """
    data.drop_duplicates(subset='text', keep="first", inplace=True)
    data.reset_index(drop=True, inplace=True)


def remove_extra_space(text):
    """
    Remove extra white spaces an keep just a single space

    Parameters
    ----------
        text : str
            text to be processed

    Returns
    -------
        clean_text : str
                clean text with removed extra white spaces

    """
    clean_text = ' '.join(text.strip().split())
    return clean_text


def remove_repeated_characters(text):
    """
    Remove repeated characters (>2) in words to max limit of 2

    Parameters
    ----------
        text : str
            text to be processed

    Returns
    -------
        clean_text : str
                cleaned text with removed repeated chars

    """
    pattern = re.compile(r'(.)\1+')
    clean_text = pattern.sub(r'\1\1', text)
    return clean_text


def remove_url(text, replace_with=''):
    """
    Remove urls from text and replace it with a special string

    Parameters
    ----------
        text : str
            text to be processed
        replace_with : str
            The string that will be used to replace the url, default <URL>

    Returns
    -------
        clean_text : str
                text with URLs replaced

    """
    pattern = re.compile(r'((www\.[\S]+)|(https?://[\S]+))')
    clean_text = pattern.sub(replace_with, text)
    return clean_text


def remove_tags(text, replace_with='<TAG>'):
    """
    Remove tags from text and replace it with special string

    Parameters
    ----------
        text : str
            text to be processed
        replace_with : str
            The string that will be used to replace the tags, default <TAG>

    Returns
    -------
        clean_text : str
                text with TAGs replaced

    """
    pattern = re.compile(r'@[\S]+')
    clean_text = pattern.sub(replace_with, text)
    return clean_text


def remove_emails(text, replace_with=''):
    """
    Remove emails from text and replace it with special string

    Parameters
    ----------
        text : str
            text to be processed
        replace_with : str
            The string that will be used to replace the tags, default <Email>

    Returns
    -------
        clean_text : str
                text with emails replaced

    """
    p = (r"([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
         r"{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|"
         r"\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)")
    pattern = re.compile(p)
    clean_text = pattern.sub(replace_with, text)
    return clean_text


def remove_hashtags(text, replace_with=''):
    """
    Remove hashtags from text and replace it with special string

    Parameters
    ----------
        text : str
            text to be processed
        replace_with : str
            The string that will be used to replace the hashtags,
            default <HASHTAG>

    Returns
    -------
        clean_text : str
            text with hashtag replaced

    """
    pattern = re.compile(r'#(\S+)')
    clean_text = pattern.sub(replace_with, text)
    return clean_text


def remove_htmltags(text):
    """
    Remove htmltags from text

    Parameters
    ----------
        text : str
            text to be processed

    Returns
    -------
        clean_text : str
            text with htmltags removed

    """
    pattern = re.compile(r'<[^>]+>')
    clean_text = pattern.sub('', text)
    return clean_text


def remove_spec_char(data):
    """
    Update the data by remove a set of special characters
    from the provided dataframe text column

    Parameters
    ----------
        data : dataframe
            data containing text column

    """
    special_char = [",", ":", "\\", "=", "&", ";", "÷",
                    "×", "*", "#", "~", "{", "(", "-", "|",
                    "`", "^", "@", "+", "[", "]", ")",
                    "}", "%", "£", "¦", "?", "ـ",
                    "،", "$", "/", "...", "..", ".", "_",
                    "---", "--", "^([0-9]+$)", "!", "٪", "•",
                    "؟", "،", "؛", "٫", "٬", "٠",
                    "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩",
                    "’", "…", "<<", "<", ">>", ">", "”", "“", "\"",
                    "–", "‘", "»", "«", "_"]
    for remove in map(lambda r: re.compile(re.escape(r)), special_char):
        data.reviews = data.reviews.str.replace(remove, " ")


def cleaning_fn(data,
                verbose=True,
                drop_nan=True,
                rmv_duplicated_text=False,
                rmv_extra_space=True,
                rmv_repeated_characters=True,
                rmv_htmltags=True,
                rmv_url=True,
                rmv_tags=True,
                rmv_emails=True,
                rmv_hashtags=True,
                cleaned_filepath=None):
    """
    Apply the basic cleaning for the whole dataset

    Parameters
    ----------
        data : pandas' dataframe
            data to be cleaned, it must contain the text column
        verbose : bool
            either to provide additional details about
            the process or not. Default True
        drop_nan : bool
            either to remove nan values or not. Default True
        rmv_duplicated_text :bool
            either to remove duplicated text or not. Default True
        rmv_extra_space : bool
            either to remove extra spaces or not. Default True
        rmv_repeated_characters: bool
            either to remove repeated characters or not. Default True
        rmv_htmltags : bool
            either to remove htmltags or not. Default True
        rmv_url : bool
            either to remove urls or not. Default True
        rmv_tags : bool
            either to remove tags or not. Default True
        rmv_emails:
            either to remove emails or not. Default True
        rmv_hashtags:
            either to remove hashtags or not. Default True
        cleaned_filepath : str
            the file path of the cleaned data. Default True

    """
    if drop_nan:
        if verbose:
            print('#' * 10, 'Step - Remove nan values')
        data.dropna(inplace=True)

    if rmv_duplicated_text:
        if verbose:
            print('#' * 10, 'Step - Remove duplicated text')
        remove_duplicated_text(data)

    if rmv_extra_space:
        if verbose:
            print('#' * 10, 'Step - Remove extra spaces')
        data.text = data.text.map(lambda x: remove_extra_space(x))

    if rmv_repeated_characters:
        if verbose:
            print('#' * 10, 'Step - Remove repeated characters')
        data.text = data.text.map(lambda x: remove_repeated_characters(x))

    if rmv_htmltags:
        if verbose:
            print('#' * 10, 'Step - Remove HTMLTAGs')
        data.text = data.text.map(lambda x: remove_htmltags(x))

    if rmv_url:
        if verbose:
            print('#' * 10, 'Step - Remove URLs')
        data.text = data.text.map(lambda x: remove_url(x))

    if rmv_tags:
        if verbose:
            print('#' * 10, 'Step - Remove TAGs')
        data.text = data.text.map(lambda x: remove_tags(x))

    if rmv_emails:
        if verbose:
            print('#' * 10, 'Step - Remove EMAILs')
        data.text = data.text.map(lambda x: remove_emails(x))

    if rmv_hashtags:
        if verbose:
            print('#' * 10, 'Step - Remove HASHTAGs')
        data.text = data.text.map(lambda x: remove_hashtags(x))

    if cleaned_filepath is not None:
        if verbose:
            print('#' * 10, 'Step - Save the cleaned data')
        data.to_csv(cleaned_filepath, index=False)
    else:
        return data
