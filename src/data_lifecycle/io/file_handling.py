from pathlib import Path
from typing import Any, Optional, Union

def open_file(file_path: Union[str, Path], mode: str='r', encoding: Optional[str]=None, **kwargs: Any) -> Any:
    """
    Open a file and return a file object or parsed data structure.

    This function handles opening files with various modes and encodings,
    returning either a raw file object or a parsed data structure depending
    on the file type and specified parameters. It supports common file formats
    and provides a unified interface for file access across the system.

    Args:
        file_path (Union[str, Path]): Path to the file to be opened.
        mode (str): File access mode ('r', 'w', 'a', etc.). Defaults to 'r'.
        encoding (Optional[str]): Text encoding to use. If None, uses system default.
        **kwargs (Any): Additional keyword arguments passed to the underlying file handler.

    Returns:
        Any: File object or parsed data structure depending on file type and mode.

    Raises:
        FileNotFoundError: If the specified file does not exist and mode is not 'w' or 'a'.
        PermissionError: If there are insufficient permissions to access the file.
        UnicodeDecodeError: If text decoding fails with the specified encoding.
    """
    path = Path(file_path)
    if 'w' in mode or 'a' in mode:
        return open(path, mode=mode, encoding=encoding, **kwargs)
    if not path.exists():
        raise FileNotFoundError(f"No such file: '{file_path}'")
    suffix = path.suffix.lower()
    if 'b' in mode:
        return open(path, mode=mode, **kwargs)
    try:
        open_kwargs = {k: v for (k, v) in kwargs.items() if k not in ['parse_json', 'parse_csv', 'return_raw']}
        file_obj = open(path, mode=mode, encoding=encoding, **open_kwargs)
        if kwargs.get('return_raw', False):
            return file_obj
        if suffix == '.json':
            if kwargs.get('parse_json', True):
                import json
                try:
                    content = json.load(file_obj)
                    file_obj.close()

                    class JsonContentWrapper:

                        def __init__(self, content):
                            self.content = content
                            self._closed = False

                        def read(self):
                            if self._closed:
                                raise ValueError('I/O operation on closed file.')
                            import json
                            return json.dumps(self.content)

                        def close(self):
                            self._closed = True

                        def __enter__(self):
                            return self

                        def __exit__(self, exc_type, exc_val, exc_tb):
                            self.close()
                            return False

                        @property
                        def parsed_content(self):
                            return self.content
                    return JsonContentWrapper(content)
                except Exception:
                    file_obj.close()
                    raise
            else:
                return file_obj
        elif suffix == '.csv':
            import csv
            if kwargs.get('parse_csv', False):
                content = list(csv.DictReader(file_obj))
                file_obj.close()

                class CsvContentWrapper:

                    def __init__(self, content):
                        self.content = content
                        self._closed = False

                    def read(self):
                        if self._closed:
                            raise ValueError('I/O operation on closed file.')
                        import csv
                        import io
                        if not self.content:
                            return ''
                        output = io.StringIO()
                        writer = csv.DictWriter(output, fieldnames=self.content[0].keys())
                        writer.writeheader()
                        writer.writerows(self.content)
                        return output.getvalue()

                    def close(self):
                        self._closed = True

                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc_val, exc_tb):
                        self.close()
                        return False

                    @property
                    def parsed_content(self):
                        return self.content
                return CsvContentWrapper(content)
            else:
                return file_obj
        else:
            return file_obj
    except FileNotFoundError:
        raise
    except PermissionError:
        raise
    except UnicodeDecodeError:
        raise
    except Exception as e:
        if 'file_obj' in locals():
            try:
                file_obj.close()
            except:
                pass
        raise IOError(f"Error processing file '{file_path}': {str(e)}")

def write_file(file_path: Union[str, Path], data: Union[str, bytes], mode: str='w') -> None:
    """
    Write data to a file, creating parent directories if needed.

    This function handles writing string or binary data to files,
    automatically creating any necessary parent directories. It
    ensures proper file handling and resource cleanup.

    Args:
        file_path (Union[str, Path]): Path where the file should be written.
        data (Union[str, bytes]): Data to write to the file. String for text mode,
                                 bytes for binary mode.
        mode (str): File write mode. Supports 'w' for text mode and 'wb' for binary mode.
                   Defaults to 'w'.

    Returns:
        None

    Raises:
        PermissionError: If there are insufficient permissions to write to the file location.
        OSError: If there are issues creating parent directories or writing the file.
        ValueError: If an unsupported mode is specified or data type doesn't match mode.
    """
    if mode not in ('w', 'wb'):
        raise ValueError(f"Unsupported mode '{mode}'. Only 'w' and 'wb' are supported.")
    if mode == 'w' and (not isinstance(data, str)):
        raise ValueError("Data must be a string when using text mode ('w').")
    if mode == 'wb' and (not isinstance(data, bytes)):
        raise ValueError("Data must be bytes when using binary mode ('wb').")
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied when creating directories for '{path}': {str(e)}")
    except OSError as e:
        raise OSError(f"Error creating directories for '{path}': {str(e)}")
    try:
        with open(path, mode=mode) as f:
            f.write(data)
    except PermissionError as e:
        raise PermissionError(f"Permission denied when writing to '{path}': {str(e)}")
    except OSError as e:
        raise OSError(f"Error writing to '{path}': {str(e)}")