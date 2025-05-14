import datetime

def is_within_24_hours(post_time):
    now = datetime.datetime.utcnow()
    difference = now - post_time
    return difference.total_seconds() <= 86400  # 24 hours in seconds
