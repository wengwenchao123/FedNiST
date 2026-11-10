import json
import time
def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    for attempt in range(50):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)