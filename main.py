import time
from src.transcriber import Transcriber


transcriber = Transcriber()
transcriber.start()

last_ts = None

print("🧪 Listening... Press Ctrl+C to stop.\n")
try:
    while True:
        if not transcriber.transcription_queue.empty():
            ts, text = transcriber.transcription_queue.get()
            if last_ts is None or ts > last_ts:
                print(f"🆕 [{ts.strftime('%H:%M:%S')}] Transcription: {text}")
                last_ts = ts
        time.sleep(1)
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")

'''from src.bot import Bot

bot = Bot()
bot.run()'''