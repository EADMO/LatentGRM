"""Recover a partially written final line before appending evaluation results."""
import json

def read_complete_rows(path):
    rows=[]
    with path.open('rb+') as stream:
        while True:
            start=stream.tell();line=stream.readline()
            if not line: break
            if not line.strip(): continue
            try: row=json.loads(line)
            except (json.JSONDecodeError,UnicodeDecodeError):
                if stream.read().strip(): raise ValueError(f'Malformed non-tail result in {path}')
                stream.seek(start);stream.truncate();break
            if not isinstance(row,dict): raise ValueError(f'Non-object result in {path}')
            rows.append(row)
            if not line.endswith(b'\n'):
                stream.write(b'\n')
                break
    return rows
