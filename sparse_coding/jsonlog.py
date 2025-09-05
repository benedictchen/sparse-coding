import json, sys, time
def log(event: str, **fields):
    rec = {"ts": time.time(), "event": event}
    rec.update(fields)
    sys.stdout.write(json.dumps(rec) + "\n")
    sys.stdout.flush()
