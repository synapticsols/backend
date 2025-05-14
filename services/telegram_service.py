import datetime
from urllib.parse import urlparse
from telethon import TelegramClient
from telethon.errors import ChannelInvalidError
from telethon.tl.functions.messages import GetHistoryRequest
from utils.time_utils import is_within_24_hours

# Your Telegram API credentials
API_ID = 26134140
API_HASH = 'fad48d6d9e50c6ac0091be337c9f6a0a'

# Session name will be saved locally
client = TelegramClient('session_name', API_ID, API_HASH)

async def fetch_telegram_messages(link: str):
    # Extract username from the link
    parsed = urlparse(link)
    username = parsed.path.strip('/')
    if not username:
        raise Exception(" Invalid Telegram link")

    await client.start()

    try:
        entity = await client.get_entity(username)
        messages = await client(GetHistoryRequest(
            peer=entity,
            limit=50,
            offset_date=None,
            offset_id=0,
            max_id=0,
            min_id=0,
            add_offset=0,
            hash=0
        ))

        filtered = []
        for msg in messages.messages:
            if (
                msg.date and 
                msg.message and 
                is_within_24_hours(msg.date.replace(tzinfo=None))
            ):
                filtered.append({
                    "content": msg.message,
                    "timestamp": msg.date.isoformat()
                })

        return {"source": "telegram", "posts": filtered}

    except ChannelInvalidError:
        raise Exception("Could not access the Telegram channel (maybe private or invalid link).")
    except Exception as e:
        raise Exception(f"Error fetching Telegram messages: {e}")
    finally:
        await client.disconnect()
